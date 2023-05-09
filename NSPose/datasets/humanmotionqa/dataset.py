import os.path as osp

import nltk
import numpy as np
from PIL import Image

import jacinle.io as io

from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jactorch.data.dataset import FilterableDatasetUnwrapped, FilterableDatasetView

from datasets.humanmotionqa.utils import nsclseq_to_nscltree, nsclseq_to_nsclqsseq, nscltree_to_nsclqstree, program_to_nsclseq

import numpy as np
import math



logger = get_logger(__file__)

__all__ = ['NSTrajDataset']
        

class NSTrajDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, data_dir, data_split_file, split, no_gt_segments, num_frames_per_seg, overlapping_frames, max_frames=150):
        super().__init__()

        self.labels_json = osp.join(data_dir, 'motion_concepts.json')
        self.questions_json = osp.join(data_dir, 'questions.json')
        self.joints_root = osp.join(data_dir, 'motion_sequences')
        
        self.labels = io.load_json(self.labels_json)
        self.questions = io.load_json(self.questions_json)
        self.split_question_ids = io.load_json(data_split_file)[split]

        self.max_frames = max_frames
        self.no_gt_segments = no_gt_segments
        self.num_frames_per_seg = num_frames_per_seg
        self.overlapping_frames = overlapping_frames
    
    def _get_metainfo(self, index):
        question = self.questions[self.split_question_ids[index]]

        # program section
        has_program = False
        if 'program_nsclseq' in question:
            question['program_raw'] = question['program_nsclseq']
            question['program_seq'] = question['program_nsclseq']
            has_program = True
        elif 'program' in question:
            question['program_raw'] = question['program']
            question['program_seq'] = program_to_nsclseq(question['program'])
            has_program = True

        if has_program:
            question['program_tree'] = nsclseq_to_nscltree(question['program_seq'])
            question['program_qsseq'] = nsclseq_to_nsclqsseq(question['program_seq'])
            question['program_qstree'] = nscltree_to_nsclqstree(question['program_tree'])

        return question
    
    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        question = self.questions[self.split_question_ids[index]]

        if 'program_raw' in metainfo:
            feed_dict.program_raw = metainfo.program_raw
            feed_dict.program_seq = metainfo.program_seq
            feed_dict.program_tree = metainfo.program_tree
            feed_dict.program_qsseq = metainfo.program_qsseq
            feed_dict.program_qstree = metainfo.program_qstree

        feed_dict.answer = question['answer']
        feed_dict.query_type = question['query_type']
        feed_dict.relation_type = question['relation_type']
        feed_dict.segment_boundaries = []
        feed_dict.question_text = question['question']
        feed_dict.question_id = question['question_id']

        # process joints
        id_name = 'babel_id'
        motion_id = question[id_name]
        num_segments = len(self.labels[motion_id])
        feed_dict.babel_id = motion_id

        joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
        # change shape of joints to match model
        joints = joints[:, :, :, np.newaxis] # T, V, C, M
        joints = joints.transpose(2, 0, 1, 3) # C, T, V, M

        # label info
        labels_frame_info = self.labels[motion_id]

        if 'filter_answer_0' in question:
            filter_segment = labels_frame_info[question['filter_answer_0']]
            if filter_segment['end_f'] > np.shape(joints)[1]:
                filter_segment['end_f'] = np.shape(joints)[1]
            feed_dict.filter_boundaries = [(filter_segment['start_f'], filter_segment['end_f'])]
            if 'filter_answer_1' in question:
                filter_segment = labels_frame_info[question['filter_answer_1']]
                if filter_segment['end_f'] > np.shape(joints)[1]:
                    filter_segment['end_f'] = np.shape(joints)[1]
                feed_dict.filter_boundaries.append((filter_segment['start_f'], filter_segment['end_f']))

        if not self.no_gt_segments:
            joints_combined = np.zeros((num_segments, 3, self.max_frames, 22, 1), dtype=np.float32) # num_segs, C, T, V, M

            for seg_i, seg in enumerate(labels_frame_info):
                if seg['end_f'] > np.shape(joints)[1]: # end frame can be slightly off
                    seg['end_f'] = np.shape(joints)[1]
                num_frames = seg['end_f'] - seg['start_f']
                
                if num_frames > self.max_frames: # clip segments to max_frames
                    num_frames = self.max_frames
                
                joints_combined[seg_i, :, :num_frames, :, :] = joints[:, seg['start_f']: seg['start_f'] + num_frames, :, :]

                feed_dict.segment_boundaries.append((seg['start_f'], (seg['start_f'] + num_frames)))
            
            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments
        else:
            total_num_frames = np.shape(joints)[1]
            num_segments = math.ceil(total_num_frames / self.num_frames_per_seg)
            feed_dict['info'] = []
            joints_combined = np.zeros((num_segments, 3, self.num_frames_per_seg + self.overlapping_frames*2, 22, 1), dtype=np.float32) # num_segs, C, T, V, M
            for i in range(num_segments):
                start_f = i * self.num_frames_per_seg
                end_f = (i + 1) * self.num_frames_per_seg
                if end_f > total_num_frames: end_f = total_num_frames

                missing_before_context = self.overlapping_frames - start_f if start_f < self.overlapping_frames else 0
                existing_after_context = total_num_frames - end_f
                if existing_after_context > self.overlapping_frames: existing_after_context = self.overlapping_frames
                
                joints_combined[i, :, missing_before_context:self.overlapping_frames+(end_f - start_f)+existing_after_context, :, :] = joints[:, start_f - (self.overlapping_frames - missing_before_context):end_f + existing_after_context, :, :]

                feed_dict.segment_boundaries.append((start_f - (self.overlapping_frames - missing_before_context), end_f + existing_after_context))
            
            feed_dict.joints = joints_combined
            feed_dict.num_segs = num_segments
            
        return feed_dict.raw()
    
    def __len__(self):
        return len(self.split_question_ids)

    # get the maximum number of segment in a single sequence across all samples (used for linear temporal projection layer)
    def get_max_num_segments(self):
        max_num_segments = 0
        for _, question in self.questions.items():
            motion_id = question['babel_id']
            if not self.no_gt_segments:
                num_segments = len(self.labels[motion_id])
            else:
                joints = np.load(osp.join(self.joints_root, motion_id, 'joints.npy')) # T, V, C
                total_num_frames = np.shape(joints)[0]
                num_segments = math.ceil(total_num_frames / self.num_frames_per_seg)

            if num_segments > max_num_segments:
                max_num_segments = num_segments
        return max_num_segments
            
class NSTrajDatasetFilterableView(FilterableDatasetView):
    def filter_questions(self, allowed):
        def filt(question):
            return question['query_type'] in allowed
            
        return self.filter(filt, 'filter-question-type[allowed={{{}}}]'.format(','.join(list(allowed))))

    def get_max_num_segments(self):
        return self.owner_dataset.get_max_num_segments()

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'joints': 'concat',
            'answer': 'skip',
            'segment_boundaries': 'skip',
            'filter_boundaries': 'skip',

            'program_raw': 'skip',
            'program_seq': 'skip',
            'program_tree': 'skip',
            'program_qsseq': 'skip',
            'program_qstree': 'skip',
        }

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide))
    
def NSTrajDataset(*args, **kwargs):
    return NSTrajDatasetFilterableView(NSTrajDatasetUnwrapped(*args, **kwargs))