import argparse
import json
import os.path as osp
import copy
import numpy as np
from collections import defaultdict
import random
import copy

np.random.seed(22)
random.seed(22)

text_templates = {"filter_action": "they #",
                  "filter_direction": "they move #",
                  "filter_body_part": "they use their #",
                  "query_action": "What action does the person do",
                  "query_direction": "What direction does the person move",
                  "query_body_part": "What body part does the person use",
                 }
MIN_QUESTIONS_PER_QUESTION_TYPE = 4

def add_query_attributes(question_temp, query_seg, seg_motion_concepts, used_concepts, questions, input_step=1, question_text_template='', relation_type='no_relation'):
    for attribute, concepts in seg_motion_concepts['attributes'].items():
        for concept in concepts:
            if concept in used_concepts[query_seg]:
                continue
            question = copy.deepcopy(question_temp)
            query_type = f'query_{attribute}'
            question['query_type'] = query_type
            question['relation_type'] = relation_type
            question['program'].append({'inputs': [input_step], 'function': query_type, 'value_inputs': []})
            question['question'] = text_templates[query_type].replace('#', concept) + ' ' + question_text_template
            question['answer'] = concept

            question_id = f'{len(questions):06d}'
            question['question_id'] = question_id
            questions[question_id] = question

def add_between_relations(question_temp, previous_relation, query_seg, seq_motion_concepts, used_concepts, questions, question_text_template, concept_counts, input_step=2):
    if previous_relation == 'before':
        filter_seg_2 = query_seg - 1
        relation_type = 'after'
    else:
        filter_seg_2 = query_seg + 1
        relation_type = 'before'
    
    if filter_seg_2 == -1 or filter_seg_2 == len(seq_motion_concepts): return

    # remove transitions
    if 'action' in seq_motion_concepts[filter_seg_2]['attributes'] and 'transition' in seq_motion_concepts[filter_seg_2]['attributes']['action']:
        if relation_type == 'before':
            filter_seg_2 = filter_seg_2 + 1 # adding one instead of subtracting because k is for filter not relation
        else:
            filter_seg_2 = filter_seg_2 - 1
        if filter_seg_2 == -1 or filter_seg_2 == len(seq_motion_concepts): return
    
    for attribute, concepts in seq_motion_concepts[filter_seg_2]['attributes'].items():
        for concept in concepts:
            used_concepts_copy = copy.deepcopy(used_concepts)
            if concept_counts[attribute][concept] > 1: continue # attribute must be unique
            question = copy.deepcopy(question_temp)
            question['program'].extend([{'inputs': [0], 'function': f'filter_{attribute}', 'value_inputs': [concept]},
                                       {'inputs': [input_step+1], 'function': 'relate', 'value_inputs': [relation_type]},
                                       {'inputs': [input_step, input_step+2], 'function': 'intersect', 'value_inputs': []}])
            question['filter_answer_1'] = filter_seg_2
            used_concepts_copy[filter_seg_2].append(concept)

            add_query_attributes(question, query_seg, seq_motion_concepts[query_seg], used_concepts_copy, questions, input_step=input_step+3, question_text_template=question_text_template[:-1] + ' and ' + relation_type +  ' ' + text_templates[f'filter_{attribute}'].replace('#', concept) + '?', relation_type='in_between')

def add_temporal_relations(question_temp, filter_seg, seq_motion_concepts, used_concepts, questions, question_text_template, concept_counts, input_step=1):
    for query_seg in [filter_seg - 1, filter_seg + 1]: # before and after relations
        if query_seg == -1 or query_seg == len(seq_motion_concepts): continue
        if query_seg < filter_seg:
            relation_type = 'before'
        else:
            relation_type = 'after'
        question = copy.deepcopy(question_temp)
        question['program'].append({'inputs': [input_step], 'function': 'relate', 'value_inputs': [relation_type]})

        # remove transitions
        if 'action' in seq_motion_concepts[query_seg]['attributes'] and 'transition' in seq_motion_concepts[query_seg]['attributes']['action']:
            if relation_type == 'before':
                query_seg = query_seg - 1
            else:
                query_seg = query_seg + 1
            if query_seg == -1 or query_seg == len(seq_motion_concepts): continue
        
        add_query_attributes(question, query_seg, seq_motion_concepts[query_seg], used_concepts, questions, input_step=2, question_text_template=relation_type + ' ' + question_text_template, relation_type=relation_type)

        add_between_relations(question, relation_type, query_seg, seq_motion_concepts, used_concepts, questions, question_text_template=relation_type + ' ' + question_text_template, concept_counts=concept_counts, input_step=input_step+1)

def generate_all_possible_questions(id, filter_seg, attribute, concept, seq_motion_concepts, concept_counts, questions):
    used_concepts = defaultdict(list) # used to not query same concept that is filtered

    question = {}
    question['babel_id'] = id
    question['program'] = [{'inputs': [], 'function': 'scene', 'value_inputs': []}, 
                                           {'inputs': [0], 'function': f'filter_{attribute}', 'value_inputs': [concept]}]
    question['filter_answer_0'] = filter_seg
    used_concepts[filter_seg].append(concept)

    # no temporal relations
    add_query_attributes(question, filter_seg, seq_motion_concepts[filter_seg], used_concepts, questions, question_text_template='while ' + text_templates[f'filter_{attribute}'].replace('#', concept) + '?')

    # with temporal relations
    add_temporal_relations(question, filter_seg, seq_motion_concepts, used_concepts, questions, text_templates[f'filter_{attribute}'].replace('#', concept) + '?', concept_counts)


def get_seq_concept_counts(seq_motion_concepts):
    concept_counts = defaultdict(lambda: defaultdict(int))
    for seg_motion_concepts in seq_motion_concepts:
        for attribute, concepts in seg_motion_concepts['attributes'].items():
            for concept in concepts:
                concept_counts[attribute][concept] += 1
    return concept_counts


def questions_by_babel_id(questions):
    babel_id_questions = defaultdict(list)

    for question_id, question in questions.items():
        babel_id = question['babel_id']
        babel_id_questions[babel_id].append(question_id)
    
    return babel_id_questions

def create_questions(motion_concepts, data_split_file):
    questions = {}

    for id in motion_concepts:
        concept_counts = get_seq_concept_counts(motion_concepts[id])

        for seg_i, seg_motion_concepts in enumerate(motion_concepts[id]):
            for attribute, concepts in seg_motion_concepts['attributes'].items():
                for concept in concepts:
                    if concept_counts[attribute][concept] > 1: continue # attribute must be unique
                    if concept == 'transition': continue

                    generate_all_possible_questions(id, seg_i, attribute, concept, motion_concepts[id], concept_counts, questions)

    if data_split_file is None:
        babel_id_questions = questions_by_babel_id(questions)
        remove_redundant_between_questions(questions, babel_id_questions)
        remove_questions_with_infrequent_concepts(questions)
        questions, babel_id_questions = balance_co_occurances(questions)
        split_question_ids = split_dataset(questions, babel_id_questions)
        split_question_ids = select_questions_with_common_concepts(questions, split_question_ids)
        with open('split_question_ids.json', 'w') as f:
            json.dump(split_question_ids, f, indent=4)
    else:
        split_question_ids = json.load(open(data_split_file))

    for split, question_ids in split_question_ids.items():
        split_babel_ids = set()
        for question_id in question_ids:
            split_babel_ids.add(questions[question_id]['babel_id'])
        print(f'{split} set contains {len(question_ids)} questions from {len(split_babel_ids)} motion sequences.')

    final_questions = {question_id: questions[question_id] for question_id in [id for split_ids in split_question_ids.values() for id in split_ids]}
    print(f'In total, generated {len(final_questions)} questions from {len(set([final_questions[question_id]["babel_id"] for question_id in final_questions]))} motion sequences.')
    return final_questions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of BABEL-QA dataset')
    parser.add_argument('--data_split_file', type=str, default=None, required=False, help='Location of train, val, and test question id splits')
    args = parser.parse_args()

    motion_concepts = json.load(open(osp.join(args.data_dir, 'motion_concepts.json')))

    questions = create_questions(motion_concepts, args.data_split_file)

    with open(osp.join(args.data_dir, 'questions.json'), 'w') as f:
        json.dump(questions, f,  indent=4)

    print(f'Questions saved at {osp.join(args.data_dir, "questions.json")}')

'''
The below functions are used to remove unwanted questions and balance co-occurances. 
Since we provide a training split file, how should not have to run these functions.
'''

def get_concept_intersection(train_counts, val_counts, test_counts):
    train_concepts = set(train_counts.keys())
    val_concepts = set(val_counts.keys())
    test_concepts = set(test_counts.keys())

    return train_concepts.intersection(val_concepts, test_concepts)

def all_question_concepts_are_allowable(question, filter_concept_intersection, query_concept_intersection):
    concepts_allowable = True

    for block in question['program']:
        if 'filter' in block['function']:
            filter_concept = block['value_inputs'][0]
            if filter_concept not in filter_concept_intersection: concepts_allowable = False
            if filter_concept not in query_concept_intersection: concepts_allowable = False # don't want concepts in filter that aren't query answers

    if question['answer'] not in query_concept_intersection: concepts_allowable = False

    return concepts_allowable

# only selects questions that have concepts existing in all three splits, for both filter and query
def select_questions_with_common_concepts(questions, split_question_ids):
    train_questions = {question_id: questions[question_id] for question_id in split_question_ids['train']}
    val_questions = {question_id: questions[question_id] for question_id in split_question_ids['val']}
    test_questions = {question_id: questions[question_id] for question_id in split_question_ids['test']}

    train_filter_concept_counts = get_dataset_concept_counts(train_questions)
    train_query_concept_counts = get_dataset_concept_counts(train_questions, step='query')
    val_filter_concept_counts = get_dataset_concept_counts(val_questions)
    val_query_concept_counts = get_dataset_concept_counts(val_questions, step='query')
    test_filter_concept_counts = get_dataset_concept_counts(test_questions)
    test_query_concept_counts = get_dataset_concept_counts(test_questions, step='query')

    filter_concept_intersection = get_concept_intersection(train_filter_concept_counts, val_filter_concept_counts, test_filter_concept_counts)
    query_concept_intersection = get_concept_intersection(train_query_concept_counts, val_query_concept_counts, test_query_concept_counts)

    for split_question_ids in [train_questions, val_questions, test_questions]:
        remove_q_ids = []
        for question_id in split_question_ids:
            if not all_question_concepts_are_allowable(questions[question_id], filter_concept_intersection, query_concept_intersection):
                remove_q_ids.append(question_id)
        for question_id in remove_q_ids:
            split_question_ids.pop(question_id)
    
    return {'train': list(train_questions.keys()), 'val': list(val_questions.keys()), 'test': list(test_questions.keys())}

def split_dataset(questions, babel_id_questions):
    split_question_ids = defaultdict(list)

    train_babel_ids = []
    val_babel_ids = []
    test_babel_ids = []
    used_babel_ids = []

    # assign each split MIN_QUESTIONS_PER_QUESTION_TYPE question of each specific question type
    query_type_specifics = defaultdict(list)
    for _, question in questions.items():
        query_type_specifics[f'{question["query_type"]}_{question["relation_type"]}'].append(question['babel_id'])
    for query_type_specific in sorted(query_type_specifics, key=lambda k: len(query_type_specifics[k])):
        query_type_specific_babel_ids = query_type_specifics[query_type_specific]
        if query_type_specific == 'query_body_part_in_between':
            query_type_specific_babel_ids_unique = dict.fromkeys(query_type_specific_babel_ids) # deterministic set ordering
            for babel_id in used_babel_ids:
                if babel_id in query_type_specific_babel_ids_unique: del query_type_specific_babel_ids_unique[babel_id]
            query_type_specific_babel_ids_unique = list(query_type_specific_babel_ids_unique)
            np.random.shuffle(query_type_specific_babel_ids_unique)
            train_babel_ids.extend(query_type_specific_babel_ids_unique[:MIN_QUESTIONS_PER_QUESTION_TYPE])
            val_babel_ids.extend(query_type_specific_babel_ids_unique[MIN_QUESTIONS_PER_QUESTION_TYPE:2*MIN_QUESTIONS_PER_QUESTION_TYPE])
            test_babel_ids.extend(query_type_specific_babel_ids_unique[2*MIN_QUESTIONS_PER_QUESTION_TYPE:3*MIN_QUESTIONS_PER_QUESTION_TYPE])
            used_babel_ids.extend(query_type_specific_babel_ids_unique[:3*MIN_QUESTIONS_PER_QUESTION_TYPE])

    train_data_percentage = 0.7
    val_data_percentage = 0.15
    test_data_percentage = 0.15

    babel_ids = list(babel_id_questions.keys())
    for babel_id in used_babel_ids:
        babel_ids.remove(babel_id)
    np.random.shuffle(babel_ids)

    train_babel_ids.extend(babel_ids[:int(len(babel_ids)*train_data_percentage)])
    val_babel_ids.extend(babel_ids[int(len(babel_ids)*train_data_percentage):int(len(babel_ids)*train_data_percentage + len(babel_ids)*val_data_percentage)])
    test_babel_ids.extend(babel_ids[int(len(babel_ids)*train_data_percentage + len(babel_ids)*val_data_percentage):])

    for split, split_babel_ids in [('train', train_babel_ids), ('val', val_babel_ids), ('test', test_babel_ids)]:
        for babel_id in split_babel_ids:
            split_question_ids[split].extend(babel_id_questions[babel_id])
    
    assert len(set(train_babel_ids).intersection(set(val_babel_ids))) == 0 and \
           len(set(train_babel_ids).intersection(set(test_babel_ids))) == 0 and \
           len(set(val_babel_ids).intersection(set(test_babel_ids))) == 0 and \
           len(set(train_babel_ids)) + len(set(val_babel_ids)) + len(set(test_babel_ids)) == len(set(babel_id_questions.keys()))

    return split_question_ids


def balance_co_occurances(questions):
    original_question_count = len(questions)
    co_occurance_counts = get_co_occurance_counts(questions)
    max_allowable_co_occurance_counts = get_max_allowable_co_occurance_counts(co_occurance_counts)

    co_occurances_in_progress = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    questions_balanced = {}
    babel_id_questions = defaultdict(list)

    # put between questions first to generate more of them
    question_ids_w_between = []
    question_ids_wout_between = []
    for question_id, question in questions.items():
        if len(question['program']) > 4:
            question_ids_w_between.append(question_id)
        else:
            question_ids_wout_between.append(question_id)
    
    np.random.shuffle(question_ids_w_between)
    np.random.shuffle(question_ids_wout_between)

    question_ids_shuffled = question_ids_w_between + question_ids_wout_between

    for question_id in question_ids_shuffled:
        question = questions[question_id]
        for block in question['program']:
            if 'filter' in block['function']:
                filter_concept = block['value_inputs'][0]
                answer_attribute = question['program'][-1]['function']

                if filter_concept not in max_allowable_co_occurance_counts or answer_attribute not in max_allowable_co_occurance_counts[filter_concept]: break
                if co_occurances_in_progress[filter_concept][answer_attribute][question['answer']] < max_allowable_co_occurance_counts[filter_concept][answer_attribute][question['answer']]:
                    if len(question['program']) < 5: # if not a between question then can add, otherwise have to check second one
                        co_occurances_in_progress[filter_concept][answer_attribute][question['answer']] += 1
                        questions_balanced[question_id] = question
                        babel_id_questions[question['babel_id']].append(question_id)
                        break
                    else: # has in between, need to check both filters
                        filter_concept_first = filter_concept
                        filter_concept_second = question['program'][-4]['value_inputs'][0]
                        if filter_concept_second not in max_allowable_co_occurance_counts or answer_attribute not in max_allowable_co_occurance_counts[filter_concept_second]: break
                        if co_occurances_in_progress[filter_concept_second][answer_attribute][question['answer']] < max_allowable_co_occurance_counts[filter_concept_second][answer_attribute][question['answer']]:
                            co_occurances_in_progress[filter_concept_first][answer_attribute][question['answer']] += 1
                            co_occurances_in_progress[filter_concept_second][answer_attribute][question['answer']] += 1
                            questions_balanced[question_id] = question
                            babel_id_questions[question['babel_id']].append(question_id)
                            break
                break
    print(f'Removed {original_question_count - len(questions_balanced)} questions from balancing co-occurances.')
    return questions_balanced, babel_id_questions
    

def get_co_occurance_counts(questions):
    co_occurances = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for _, question in questions.items():
        for block in question['program']:
            if 'filter' in block['function']:
                filter_concept = block['value_inputs'][0]

                answer_attribute = question['program'][-1]['function']
                co_occurances[filter_concept][answer_attribute][question['answer']] += 1
    return co_occurances

def get_max_allowable_co_occurance_counts(co_occurance_counts, threshold=0.34): 
    max_allowable_co_occurance_counts = copy.deepcopy(co_occurance_counts)
    for filter_concept in co_occurance_counts:
        for answer_attribute in co_occurance_counts[filter_concept]:
            possible_answer_count = len(co_occurance_counts[filter_concept][answer_attribute].values())
            if 1 / possible_answer_count > threshold:
                del max_allowable_co_occurance_counts[filter_concept][answer_attribute]
                continue
            balance_in_progress = True
            while balance_in_progress:
                changes_made = False
                for answer, answer_count in max_allowable_co_occurance_counts[filter_concept][answer_attribute].items():
                    total_attribute_questions = sum(max_allowable_co_occurance_counts[filter_concept][answer_attribute].values())
                    max_allowed = int(threshold * total_attribute_questions)
                    if max_allowable_co_occurance_counts[filter_concept][answer_attribute][answer] > max_allowed:
                        max_allowable_co_occurance_counts[filter_concept][answer_attribute][answer] = max_allowed
                        changes_made = True
                if changes_made == False:
                    balance_in_progress = False
    return max_allowable_co_occurance_counts


def get_dataset_concept_counts(questions, step='filter'):
    concept_counts = defaultdict(int)
    for _, question in questions.items():
        if step == 'filter':
            for block in question['program']:
               if 'filter' in block['function']:
                   filter_concept = block['value_inputs'][0]
                   concept_counts[filter_concept] += 1
        elif step == 'query':
            concept_counts[question['answer']] += 1

    return concept_counts


def remove_questions_with_infrequent_concepts(questions, threshold=8):
    original_question_count = len(questions)
    filter_concept_counts = get_dataset_concept_counts(questions)
    query_concept_counts = get_dataset_concept_counts(questions, step='query')
    remove_q_ids = []

    for question_id, question in questions.items():

        remove_question = False
        for block in question['program']:
            if 'filter' in block['function']:
                filter_concept = block['value_inputs'][0]
                if filter_concept_counts[filter_concept] < threshold:
                    remove_question = True
                    break
        if query_concept_counts[question['answer']] < threshold:
            remove_question = True
        if remove_question: remove_q_ids.append(question_id)

    for question_id in remove_q_ids:
        questions.pop(question_id)

    print(f'Removed {original_question_count - len(questions)} questions with infrequent concepts.')



def remove_redundant_between_questions(questions, babel_id_questions):
    original_question_count = len(questions)
    for _, question_ids in babel_id_questions.items():
        skip_q_ids = []
        remove_q_ids = []

        for question_id1 in question_ids:
            q1 = questions[question_id1]
            if question_id1 in skip_q_ids or question_id1 in remove_q_ids: continue
            if len(q1['program']) == 7:
                for question_id2 in question_ids:
                    q2 = questions[question_id2]
                    if question_id1 != question_id2 and len(q2['program']) == 7:
                        # same filters with same answer signifies questions are duplicates
                        if q1['program'][1]['value_inputs'] == q2['program'][3]['value_inputs'] and \
                           q1['program'][3]['value_inputs'] == q2['program'][1]['value_inputs'] and \
                           q1['answer'] == q2['answer']:
                            if np.random.uniform() < 0.5:
                                skip_q_ids.append(question_id1)
                                remove_q_ids.append(question_id2)
                            else:
                                skip_q_ids.append(question_id2)
                                remove_q_ids.append(question_id1)
                            break
        for question_id in remove_q_ids:
            questions.pop(question_id)
            question_ids.remove(question_id)

    print(f'Removed {original_question_count - len(questions)} redundant between questions.')

if __name__ == '__main__':
    main()