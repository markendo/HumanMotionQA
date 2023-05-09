from jacinle.utils.container import GView
from models.model import make_motion_reasoning_configs, MotionReasoningModel

import numpy as np
import torch.nn as nn
import torch



configs = make_motion_reasoning_configs()

class Model(MotionReasoningModel):
    def __init__(self, no_gt_segments, temporal_operator, max_num_segments):
        self.no_gt_segments = no_gt_segments
        self.temporal_operator = temporal_operator
        self.max_num_segments = max_num_segments
        super().__init__(configs)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}
        motion_encodings = self.motion_encoder(feed_dict.joints)

        # build scene
        f_sng = []
        start_seg = 0
        for seq_num_segs in feed_dict.num_segs:
            f_sng.append([None, motion_encodings[start_seg:start_seg+seq_num_segs], None])
            start_seg += seq_num_segs
        assert start_seg == motion_encodings.size()[0]

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs)
        outputs['buffers'] = buffers
        outputs['answer'] = answers
        outputs['query_type'] = feed_dict.query_type

        # intermediate filter monitoring (not supervision)
        update_from_loss_module(monitors, outputs, self.filter_monitor(feed_dict, answers, buffers))
        # final loss
        update_from_loss_module(monitors, outputs, self.query_loss(feed_dict, answers, buffers))
        
        canonize_monitors(monitors)

        if self.training:
            if 'loss/query' in monitors and 'loss/filter' in monitors:
                loss = monitors['loss/query'] + monitors['loss/filter'] # can finetune ratio
            elif 'loss/query' in monitors:
                loss = monitors['loss/query']
            elif 'loss/filter' in monitors:
                loss = monitors['loss/filter']
            else:
                loss = 0 # happens when not using filter loss and all questions are filter related
            return loss, monitors, outputs
        else:
            outputs['monitors'] = monitors
            return outputs
            
def make_model(args, max_num_segments):
    return Model(args.no_gt_segments, args.temporal_operator, max_num_segments)

def canonize_monitors(monitors):
    for k, v in monitors.items():
        if isinstance(monitors[k], list):
            if isinstance(monitors[k][0], tuple) and len(monitors[k][0]) == 2:
                monitors[k] = sum([a * b for a, b in monitors[k]]) / max(sum([b for _, b in monitors[k]]), 1e-6)
            else:
                monitors[k] = sum(v) / max(len(v), 1e-3)
        if isinstance(monitors[k], float):
            monitors[k] = torch.tensor(monitors[k])


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)

