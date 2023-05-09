from ast import Gt
from re import A
import torch.nn as nn
import torch
import torch.nn.functional as F
from datasets.definition import gdef

class MultitaskLossBase(nn.Module):
    def __init__(self):
        super().__init__()

    def _xent_loss(self, pred, label):
        logp = F.log_softmax(pred, dim=-1)
        return -logp[label].mean()

class IntermediateFilterMonitor(MultitaskLossBase):
    def __init__(self):
        super().__init__()
    
    def forward(self, feed_dict, answers, buffers):
        monitors = {}
        outputs = {}

        for i, prog in enumerate(feed_dict.program_qsseq):
            filter_num = 0
            for block_id, block in enumerate(prog):
                if block['op'] == 'filter':
                    inter_a = buffers[i][block_id]
                    argmax = inter_a.argmax(dim=-1).item()

                    # change to middle of argmax segment intersects with gt
                    middle_seg = (feed_dict['segment_boundaries'][i][argmax][0] + feed_dict['segment_boundaries'][i][argmax][1]) / 2
                    acc =  (middle_seg >= feed_dict['filter_boundaries'][i][filter_num][0] and middle_seg <= feed_dict['filter_boundaries'][i][filter_num][1])

                    monitors.setdefault(feed_dict.program_raw[i][block_id]['function'], []).append((acc, 1))
                    filter_num += 1

        return monitors, outputs


class QueryLoss(MultitaskLossBase):
    def __init__(self):
        super().__init__()
    
    def forward(self, feed_dict, answers, buffers):
        monitors = {}
        outputs = {'answer': [], 'gt': []}

        for i, (query_type, a) in enumerate(answers):
            if query_type == 'query':
                response_query_type = gdef.qtype2atype_dict[query_type]
                gt = feed_dict['answer'][i]
                if response_query_type == 'word':
                    a, word2idx = a
                    argmax = a.argmax(dim=-1).item()
                    idx2word = {v: k for k, v in word2idx.items()}
                    gt_word = gt
                    gt = word2idx[gt]
                    loss = self._xent_loss
                    query_type = feed_dict['query_type'][i]
                    monitors.setdefault('acc/' + query_type, []).append((int(gt == argmax), 1))

                    relation_type = feed_dict.relation_type[i]
                    query_type_specific = f'{query_type}_{relation_type}'

                    monitors.setdefault('acc/' + query_type_specific, []).append((int(gt == argmax), 1))
                    
                    monitors.setdefault('acc/qa', []).append((int(gt == argmax), 1))

                    outputs['answer'].append(idx2word[argmax])
                    outputs['gt'].append(gt_word)

                    if self.training:
                        l = loss(a, gt)
                        monitors.setdefault('loss/query', []).append((l, 1))
        return monitors, outputs