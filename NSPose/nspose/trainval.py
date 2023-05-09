import time
import os.path as osp
from collections import defaultdict

import torch.backends.cudnn as cudnn
import torch.cuda as cuda

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.imp import load_source
from jacinle.utils.tqdm import tqdm_pbar
from jaclearn.mldash import MLDashClient

from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jactorch.cuda.copy import async_copy_to
from jactorch.train import TrainerEnv
from jactorch.utils.meta import as_float

import json

logger = get_logger(__file__)

parser = JacArgumentParser(description='')

parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--configs', default='', type='kv', metavar='CFGS')

parser.add_argument('--no_gt_segments', action='store_true', help='dont use gt action segments')
parser.add_argument('--temporal_operator', default='linear', choices=['linear', 'conv1d', 'shift'])
parser.add_argument('--num-frames-per-seg', default=45, type=int, metavar='N', help='number of frames per each segment')
parser.add_argument('--overlapping-frames', default=15, type=int, metavar='N', help='number of frames that each segment overlaps with one another')

# training_target and curriculum learning
parser.add_argument('--expr', default='default', metavar='DIR', help='experiment name')
parser.add_argument('--curriculum', default='off', choices=['off', 'on'])
# running mode
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--evaluate', action='store_true', help='run the validation only; used with --load or --resume')

# training hyperparameters
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of total epochs to run')
parser.add_argument('--enums-per-epoch', type=int, default=1, metavar='N', help='number of enumerations of the whole dataset per epoch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='initial learning rate')
parser.add_argument('--iters-per-epoch', type=int, default=0, metavar='N', help='number of iterations per epoch 0=one pass of the dataset (default: 0)')
parser.add_argument('--acc-grad', type=int, default=1, metavar='N', help='accumulated gradient (default: 1)')
parser.add_argument('--clip-grad', type=float, metavar='F', help='gradient clipping')
parser.add_argument('--validation-interval', type=int, default=1, metavar='N', help='validation inverval (epochs) (default: 1)')

# finetuning and snapshot
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--resume', type='checked_file', default=None, metavar='FILE', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='manual epoch number')
parser.add_argument('--save-interval', type=int, default=2, metavar='N', help='model save interval (epochs) (default: 10)')

# data related
parser.add_argument('--data-dir', required=True, type='checked_dir', metavar='DIR', help='data directory')
parser.add_argument('--data_split_file', type='checked_file', default='../BABEL-QA/split_question_ids.json', metavar='FILE', help='file that contains which questions are included in train, val, and test')

parser.add_argument('--data-workers', type=int, default=2, metavar='N', help='the num of workers that input training data')

# misc
parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--use-tb', type='bool', default=True, metavar='B', help='use tensorboard or not')
parser.add_argument('--embed', action='store_true', help='entering embed after initialization')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

args = parser.parse_args()

# filenames
args.series_name = 'nspose'
args.desc_name = escape_desc_name(args.desc)
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)

desc = load_source(args.desc)
configs = desc.configs
args.configs.apply(configs)

mldash = MLDashClient('dumps')

def main():
    args.dump_dir = ensure_path(osp.join('dumps', args.series_name, args.desc_name, args.expr, args.run_name))

    if not args.debug:
        args.ckpt_dir = ensure_path(osp.join(args.dump_dir, 'checkpoints'))
        args.meta_dir = ensure_path(osp.join(args.dump_dir, 'meta'))
        args.meta_file = osp.join(args.meta_dir, args.run_name + '.json')
        args.log_file = osp.join(args.meta_dir, args.run_name + '.log')
        args.meter_file = osp.join(args.meta_dir, args.run_name + '.meter.json')
        
        logger.critical('Writing logs to file: "{}".'.format(args.log_file))
        set_output_file(args.log_file)

        logger.critical('Writing metainfo to file: "{}".'.format(args.meta_file))
        with open(args.meta_file, 'w') as f:
            f.write(dump_metainfo(args=args.__dict__, configs=configs))

        # Initialize the tensorboard.
        if args.use_tb:
            args.tb_dir = ensure_path(osp.join(args.dump_dir, 'tensorboard'))

    logger.critical('Building the data set.')
    
    from datasets.definition import set_global_definition
    from datasets.humanmotionqa.definition import HumanMotionQADefinition
    set_global_definition(HumanMotionQADefinition())

    train_dataset = build_human_motion_dataset(args.data_dir, args.data_split_file, 'train', no_gt_segments=args.no_gt_segments, num_frames_per_seg=args.num_frames_per_seg, overlapping_frames=args.overlapping_frames)
    val_dataset = build_human_motion_dataset(args.data_dir, args.data_split_file, 'val', no_gt_segments=args.no_gt_segments,  num_frames_per_seg=args.num_frames_per_seg, overlapping_frames=args.overlapping_frames)
    test_dataset = build_human_motion_dataset(args.data_dir, args.data_split_file, 'test', no_gt_segments=args.no_gt_segments,  num_frames_per_seg=args.num_frames_per_seg, overlapping_frames=args.overlapping_frames)

    main_train(train_dataset, val_dataset, test_dataset)
    
    
def build_human_motion_dataset(data_dir, data_split_file, split, no_gt_segments=False, num_frames_per_seg=45, overlapping_frames=15):
    from datasets.humanmotionqa.dataset import NSTrajDataset
    dataset = NSTrajDataset(data_dir, data_split_file, split, no_gt_segments, num_frames_per_seg, overlapping_frames)
    return dataset

def main_train(train_dataset, validation_dataset, test_dataset):
    logger.critical('Building the model.')
    model = desc.make_model(args, train_dataset.get_max_num_segments())

    if args.use_gpu:
        model.cuda()
        # Use the customized data parallel if applicable.
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        # Disable the cudnn benchmark.
        cudnn.benchmark = False

    if hasattr(desc, 'make_optimizer'):
        logger.critical('Building customized optimizer.')
        optimizer = desc.make_optimizer(model, args.lr)
    else:
        from jactorch.optim import AdamW
        trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = AdamW(trainable_parameters, args.lr, weight_decay=configs.train.weight_decay)

    if args.acc_grad > 1:
        from jactorch.optim import AccumGrad
        optimizer = AccumGrad(optimizer, args.acc_grad)
        logger.warning('Use accumulated grad={:d}, effective iterations per epoch={:d}.'.format(args.acc_grad, int(args.iters_per_epoch / args.acc_grad)))

    trainer = TrainerEnv(model, optimizer)

    if args.resume:
        extra = trainer.load_checkpoint(args.resume)
        if extra:
            args.start_epoch = extra['epoch']
            logger.critical('Resume from epoch {}.'.format(args.start_epoch))
    elif args.load:
        if trainer.load_weights(args.load):
            logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))

    if args.use_tb and not args.debug:
        from jactorch.train.tb import TBLogger, TBGroupMeters
        tb_logger = TBLogger(args.tb_dir)
        meters = TBGroupMeters(tb_logger)
        logger.critical('Writing tensorboard logs to: "{}".'.format(args.tb_dir))
    else:
        from jacinle.utils.meter import GroupMeters
        meters = GroupMeters()

    if not args.debug:
        logger.critical('Writing meter logs to file: "{}".'.format(args.meter_file))

    if not args.debug:
        logger.critical('Initializing MLDash.')
        mldash.init(
            desc_name=args.series_name + '/' + args.desc_name,
            expr_name=args.expr,
            run_name=args.run_name,
            args=args,
            highlight_args=parser,
            configs=configs,
        )
        mldash.update(metainfo_file=args.meta_file, log_file=args.log_file, meter_file=args.meter_file, tb_dir=args.tb_dir)

    if args.clip_grad:
        logger.info('Registering the clip_grad hook: {}.'.format(args.clip_grad))
        def clip_grad(self, loss):
            from torch.nn.utils import clip_grad_norm_
            clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad)
        trainer.register_event('backward:after', clip_grad)

    if hasattr(desc, 'customize_trainer'):
        desc.customize_trainer(trainer)

    if args.embed:
        from IPython import embed; embed()

    logger.critical('Building the data loader.')
    validation_dataloader = validation_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    if args.evaluate:
        test_dataloader = test_dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)
        meters.reset()
        model.eval()
        validate_epoch(0, trainer, test_dataloader, meters, meter_prefix='test')
        logger.critical(meters.format_simple('Validation', {k: v for k, v in meters.avg.items() if v != 0}, compressed=False))
        meters.dump(args.meter_file)
        return meters

    curriculum_strategy = [
        (0, ['query_action']),
        (5, ['query_action', 'query_body_part', 'query_direction']),
    ]

    top_acc = 0

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        meters.reset()

        model.train()

        this_train_dataset = train_dataset
        if args.curriculum != 'off':
            for si, s in enumerate(curriculum_strategy):
                if curriculum_strategy[si][0] < epoch <= curriculum_strategy[si + 1][0]:
                    allowed_query_types = s[1]
                    if len(allowed_query_types) != 0:
                        this_train_dataset = this_train_dataset.filter_questions(allowed_query_types)

        train_dataloader = this_train_dataset.make_dataloader(args.batch_size, shuffle=True, drop_last=True, nr_workers=args.data_workers)

        for enum_id in range(args.enums_per_epoch):
            train_epoch(epoch, trainer, train_dataloader, meters)

        if epoch % args.validation_interval == 0:
            model.eval()
            val_acc = validate_epoch(epoch, trainer, validation_dataloader, meters)

        if not args.debug:
            meters.dump(args.meter_file)

        if not args.debug:
            mldash.log_metric('epoch', epoch, desc=False, expr=False)
            for key, value in meters.items():
                if key.startswith('loss') or key.startswith('validation/loss'):
                    mldash.log_metric_min(key, value.avg)
            for key, value in meters.items():
                if key.startswith('acc') or key.startswith('validation/acc'):
                    mldash.log_metric_max(key, value.avg)

        logger.critical(meters.format_simple(
            'Epoch = {}'.format(epoch),
            {k: v for k, v in meters.avg.items() if epoch % args.validation_interval == 0 or not k.startswith('validation')},
            compressed=False
        ))


        if epoch % args.save_interval == 0 and not args.debug:
            fname = osp.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
        
        if val_acc > top_acc:
            fname = osp.join(args.ckpt_dir, 'best.pth')
            trainer.save_checkpoint(fname, dict(epoch=epoch, meta_file=args.meta_file))
            top_acc = val_acc

        if epoch > int(args.epochs * 0.6):
            trainer.set_learning_rate(args.lr * 0.1)


def backward_check_nan(self, feed_dict, loss, monitors, output_dict):
    import torch
    for name, param in self.model.named_parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad.data).any().item():
            print('Caught NAN in gradient.', name)
            from IPython import embed; embed()


def train_epoch(epoch, trainer, train_dataloader, meters):
    nr_iters = args.iters_per_epoch
    if nr_iters == 0:
        nr_iters = len(train_dataloader)

    meters.update(epoch=epoch)

    trainer.trigger_event('epoch:before', trainer, epoch)
    train_iter = iter(train_dataloader)

    end = time.time()
    with tqdm_pbar(total=nr_iters) as pbar:
        for i in range(nr_iters):
            feed_dict = next(train_iter)

            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()

            loss, monitors, output_dict, extra_info = trainer.step(feed_dict, cast_tensor=False)

            step_time = time.time() - end; end = time.time()

            n = feed_dict['num_segs'].size(0) # number of sequences

            meters.update(loss=loss, n=n)
            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {}'.format(epoch),
                {k: v for k, v in meters.val.items() if not k.startswith('validation') and k != 'epoch' and k.count('/') <= 1},
                compressed=True
            ))
            pbar.update()

            end = time.time()

    trainer.trigger_event('epoch:after', trainer, epoch)


def validate_epoch(epoch, trainer, val_dataloader, meters, meter_prefix='validation'):
    end = time.time()
    out_logs = {}

    with tqdm_pbar(total=len(val_dataloader)) as pbar:
        for feed_dict in val_dataloader:
            if args.use_gpu:
                if not args.gpu_parallel:
                    feed_dict = async_copy_to(feed_dict, 0)

            data_time = time.time() - end; end = time.time()
            output_dict, extra_info = trainer.evaluate(feed_dict, cast_tensor=False)

            for i, prog in enumerate(feed_dict['program_qsseq']):
                question_id = feed_dict['question_id'][i]
                concept_boundary_preds = defaultdict(list)
                filter_concepts = []
                
                for block_id, block in enumerate(prog):
                    if block['op'] == 'filter':
                        filter_probs = output_dict['buffers'][i][block_id].cpu().numpy().tolist()
                        concept_boundary_preds['filter_probs'].append(filter_probs)
                        filter_concepts.append(block['concept'][0])
                    elif block['op'] == 'relate':
                        relation_probs = output_dict['buffers'][i][block_id].cpu().numpy().tolist()
                        concept_boundary_preds['relation_probs'].append(relation_probs)
                    elif block['op'] == 'intersect':
                        combined_relation_probs = output_dict['buffers'][i][block_id].cpu().numpy().tolist()
                        concept_boundary_preds['relation_probs'].append(combined_relation_probs)
                out_logs[question_id] = {'babel_id': feed_dict['babel_id'][i],'question_text': feed_dict['question_text'][i], 'gt': output_dict['gt'][i], 'answer': output_dict['answer'][i], 'relation_type': feed_dict['relation_type'][i], 'filter_boundaries': feed_dict['filter_boundaries'][i], 'segment_boundaries': feed_dict['segment_boundaries'][i], 'filter_concepts': filter_concepts, 'concept_boundary_preds': concept_boundary_preds}

            monitors = {meter_prefix + '/' + k: v for k, v in as_float(output_dict['monitors']).items()}
            step_time = time.time() - end; end = time.time()

            n = feed_dict['num_segs'].size(0) # number of sequences

            meters.update(monitors, n=n)
            meters.update({'time/data': data_time, 'time/step': step_time})

            if args.use_tb:
                meters.flush()

            pbar.set_description(meters.format_simple(
                'Epoch {} (validation)'.format(epoch),
                {k: v for k, v in meters.val.items() if k.startswith('validation') and k.count('/') <= 2},
                compressed=True
            ))
            pbar.update()

            end = time.time()
    
    args.out_file = osp.join(args.meta_dir, 'out.json')
    logger.critical('Writing out logs to file: "{}".'.format(args.out_file))
    with open(args.out_file, 'w') as f:
        json.dump(out_logs, f)

    return meters.avg[f'{meter_prefix}/acc/qa']

if __name__ == '__main__':
    main()