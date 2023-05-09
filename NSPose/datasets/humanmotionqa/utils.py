from collections import defaultdict
from copy import deepcopy
from datasets.definition import gdef

from jacinle.utils.tqdm import tqdm

__all__ = ['nsclseq_to_nscltree', 'nsclseq_to_nsclqsseq', 'nscltree_to_nsclqstree', 'program_to_nsclseq']

def nsclseq_to_nscltree(seq_program):
    def dfs(sblock):
        tblock = deepcopy(sblock)
        input_ids = tblock.pop('inputs')
        tblock['inputs'] = [dfs(seq_program[i]) for i in input_ids]
        return tblock

    try:
        return dfs(seq_program[-1])
    finally:
        del dfs
        
def nsclseq_to_nsclqsseq(seq_program):
    qs_seq = deepcopy(seq_program)
    cached = defaultdict(list)

    for sblock in qs_seq:
        for param_type in gdef.parameter_types:
            if param_type in sblock:
                sblock[param_type + '_idx'] = len(cached[param_type])
                sblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(sblock[param_type])

    return qs_seq

def nscltree_to_nsclqstree(tree_program):
    qs_tree = deepcopy(tree_program)
    cached = defaultdict(list)

    for tblock in iter_nscltree(qs_tree):
        for param_type in gdef.parameter_types:
            if param_type in tblock:
                tblock[param_type + '_idx'] = len(cached[param_type])
                tblock[param_type + '_values'] = cached[param_type]
                cached[param_type].append(tblock[param_type])

    return qs_tree

def iter_nscltree(tree_program):
    yield tree_program
    for i in tree_program['inputs']:
        yield from iter_nscltree(i)
        
def get_clevr_pblock_op(block):
    if 'type' in block:
        return block['type']
    assert 'function' in block
    return block['function']

def get_clevr_op_attribute(op):
    return "_".join(op.split('_')[1:])

def program_to_nsclseq(program):
    nscl_program = list()
    mapping = dict()

    for block_id, block in enumerate(program):
        op = get_clevr_pblock_op(block)
        current = None
        if op == 'scene':
            current = dict(op='scene')
        elif op.startswith('filter'):
            concept = block['value_inputs'][0]
            last = nscl_program[mapping[block['inputs'][0]]]
            if last['op'] == 'filter':
                last['concept'].append(concept)
            else:
                current = dict(op='filter', concept=[concept])
        elif op.startswith('relate'):
            concept = block['value_inputs'][0]
            current = dict(op='relate', relational_concept=[concept])
        elif op.startswith('same'):
            attribute = get_clevr_op_attribute(op)
            current = dict(op='relate_attribute_equal', attribute=attribute)
        elif op in ('intersect', 'union'):
            current = dict(op=op)
        elif op == 'unique':
            pass  # We will ignore the unique operations.
        else:
            if op.startswith('query'):
                if block_id == len(program) - 1:
                    attribute = get_clevr_op_attribute(op)
                    current = dict(op='query', attribute=attribute)
            elif op.startswith('equal') and op != 'equal_integer':
                attribute = get_clevr_op_attribute(op)
                current = dict(op='query_attribute_equal', attribute=attribute)
            elif op == 'exist':
                current = dict(op='exist')
            elif op == 'count':
                if block_id == len(program) - 1:
                    current = dict(op='count')
            elif op == 'equal_integer':
                current = dict(op='count_equal')
            elif op == 'less_than':
                current = dict(op='count_less')
            elif op == 'greater_than':
                current = dict(op='count_greater')
            else:
                raise ValueError('Unknown CLEVR operation: {}.'.format(op))

        if current is None:
            assert len(block['inputs']) == 1
            mapping[block_id] = mapping[block['inputs'][0]]
        else:
            current['inputs'] = list(map(mapping.get, block['inputs']))

            if '_output' in block:
                current['output'] = deepcopy(block['_output'])

            nscl_program.append(current)
            mapping[block_id] = len(nscl_program) - 1

    return nscl_program