"""
Create dataset for BABEL.
"""
import argparse
import json
import os
from os.path import join as ospj
from os.path import basename as ospb
from os.path import dirname as ospd
from os.path import isfile

import numpy as np

import sys

non_ambiguous_actions = ['bow', 'cartwheel', 'catch', 'clap', 'cough', 'crawl', 'crouch', 'dribble', 'drink', 'duck', 'eat', 'fall', 'fight', 'fire gun', 'fish', 'fist bump', 'flail arms', 'flap', 'flip', 'golf', 'handstand', 'hop', 'hug', 'jog', 'juggle', 'jump', 'jump rope', 'jumping jacks', 'kick', 'kneel', 'knock', 'leap', 'lift something', 'limp', 'lunge', 'march', 'mix', 'moonwalk', 'place something', 'punch', 'rake', 'run', 'salute', 'shuffle', 'sit', 'skate', 'skip', 'sneeze', 'squat', 'stand up', 'take/pick something up', 'throw', 'tie', 'tiptoe', 'walk', 'wave', 'yawn', 'transition']
body_parts = ['left hand', 'right hand', 'left arm', 'right arm', 'left leg', 'right leg', 'left foot', 'right foot']
directions = ['left', 'right', 'forward', 'forwards', 'backwards', 'backward']

# combine certain categories
synonyms = {"backwards": ["backwards", "backward"],
            "forward": ["forwards", "forward"],
            "jump": ["jump", "hop"],
            "take/pick something up": ["take/pick something up", "lift something"]
}

discard_babel_ids = [7465, 10967] # discard sequences based on terrain interaction

FPS = 30 # all sequences processed to 30fps

def get_attributes(label):
    segment_attributes = {}
    proc_label = label['proc_label']
    
    # actions
    seg_actions = []
    for action in non_ambiguous_actions:
        if action in label['act_cat']:

            for word in synonyms:
                for synonym in synonyms[word]:
                    if action == synonym: action = word

            if action not in seg_actions:
                seg_actions.append(action)
    if len(seg_actions) >= 1:
        segment_attributes['action'] = seg_actions

    # body part
    seg_body_parts = []
    for body_part in body_parts:
        if body_part in proc_label:
            
            for word in synonyms:
                for synonym in synonyms[word]:
                    if body_part == synonym: body_part = word

            if body_part not in seg_body_parts:
                seg_body_parts.append(body_part)
    if len(seg_body_parts) >= 1:
        segment_attributes['body_part'] = seg_body_parts
   
    # direction
    seg_directions = []
    for direction in directions:
        if direction in proc_label:
            # ensure direction name not part of body part, eg., "right" hand
            valid = True
            for body_part in seg_body_parts:
                if direction in body_part: valid = False
            if not valid: continue
            
            for word in synonyms:
                for synonym in synonyms[word]:
                    if direction == synonym: direction = word
            
            if direction not in seg_directions:
                seg_directions.append(direction)
    if len(seg_directions) >= 1:
        segment_attributes['direction'] = seg_directions
    
    return segment_attributes

def overlapping_segments(ann, ann_type):
    # from https://stackoverflow.com/questions/35644301/checking-two-time-intervals-are-overlapping-or-not
    def overlap(seg1,seg2):
        for time in (seg1['start_t'], seg1['end_t']):
            if seg2['start_t'] < time < seg2['end_t']:
                return True
        else:
            return False

    num_overlaps = 0
    for i in range(len(ann['labels'])):
        for j in range(i + 1, len(ann['labels'])):
            if overlap(ann['labels'][i], ann['labels'][j]) or overlap(ann['labels'][j], ann['labels'][i]): num_overlaps += 1
    return num_overlaps

def process_sequence(params, annotation, out_labels):
    # get per frame annotations
    ann_type = 'frame_ann' if annotation['frame_ann'] is not None else 'seq_ann'

    # some seq annotations don't have start_t and end_t
    if ann_type == 'seq_ann' and 'start_t' not in annotation[ann_type]['labels']:
        if len(annotation[ann_type]['labels']) != 1: return 0
        annotation[ann_type]['labels'][0]['start_t'] = 0
        annotation[ann_type]['labels'][0]['end_t'] = annotation['dur']

    if annotation['babel_sid'] in discard_babel_ids:
        return

    num_overlaps = overlapping_segments(annotation[ann_type], ann_type)
    if num_overlaps > 0: # only process sequences that don't have any overlapping segments
        return

    segment_order = np.argsort([seg['start_t'] for seg in annotation[ann_type]['labels']])
    seq_out_labels = []
    for i in segment_order:
        seg_label = {}
        seg = annotation[ann_type]['labels'][i]

        # get attributes
        segment_attributes = get_attributes(seg)
        
        seg_start_frame, seg_end_frame = int(FPS * seg['start_t']), int(FPS * seg['end_t'])

        seg_label['start_f'] = seg_start_frame
        seg_label['end_f'] = seg_end_frame
        seg_label['attributes'] = segment_attributes
        if seg_start_frame != seg_end_frame:
            seq_out_labels.append(seg_label)

    assert annotation['babel_sid'] not in out_labels
    out_labels[annotation['babel_sid']] = seq_out_labels


def process_split(params, annotations, out_labels):
    for annotation in annotations:
        process_sequence(params, annotation, out_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--babel_root', type=str, required=True, help='Root directory of the BABEL dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of BABEL-QA dataset')
    args = parser.parse_args()
    params = vars(args)

    out_labels = {}
    for spl in ['train', 'val']:
        ann = json.load(open(ospj(params['babel_root'], f'{spl}.json')))
        annotations = [ann[sid] for sid in ann]

        process_split(params, annotations, out_labels)
    
    if not os.path.exists(params['data_dir']):
        os.makedirs(params['data_dir'])
    
    with open(ospj(params['data_dir'], 'motion_concepts.json'), 'w') as f:
        json.dump(out_labels, f, indent=4)
    
if __name__ == '__main__':
    main()