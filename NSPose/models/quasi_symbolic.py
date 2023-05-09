import torch
import torch.nn as nn
import jactorch.nn.functional as jacf
from . import concept_embedding
from datasets.definition import gdef
from jacinle.utils.enum import JacEnum
import six

import jactorch.nn as jacnn

_apply_self_mask = {'relate': True, 'relate_ae': True}

class ParameterResolutionMode(JacEnum):
    DETERMINISTIC = 'deterministic'
    PROBABILISTIC_SAMPLE = 'probabilistic_sample'
    PROBABILISTIC_ARGMAX = 'probabilistic_argmax'

def do_apply_self_mask(m):
    self_mask = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
    return m * (1 - self_mask) + (-10) * self_mask

class InferenceQuantizationMethod(JacEnum):
    NONE = 0
    STANDARD = 1
    EVERYTHING = 2


_test_quantize = InferenceQuantizationMethod.STANDARD


def set_test_quantize(mode):
    global _test_quantize
    _test_quantize = InferenceQuantizationMethod.from_string(mode)



class ProgramExecutorContext(nn.Module):
    def __init__(self, attribute_taxnomy, relation_taxnomy, features, parameter_resolution, temporal_operator, training=True, before_projection_layer=None, after_projection_layer=None, max_num_segments=50):
        super().__init__()

        self.features = features
        self.parameter_resolution = ParameterResolutionMode.from_string(parameter_resolution)

        # None, attributes, relations
        self.taxnomy = [None, attribute_taxnomy, relation_taxnomy]
        self._concept_groups_masks = [None, None, None]

        self._attribute_groups_masks = None
        self._attribute_query_masks = None
        self._attribute_query_ls_masks = None
        self._attribute_query_ls_mc_masks = None

        self.before_projection_layer = before_projection_layer
        self.after_projection_layer = after_projection_layer
        self.max_num_segments = max_num_segments

        self.temporal_operator = temporal_operator

        self.train(training)

    def filter(self, selected, group, concept_groups): # selected: inputs (10 for each segment), group: concept indx (0), concept_values: action eg place something
        if group is None:
            return selected
        mask = self._get_concept_groups_masks(concept_groups, 1) # gets logits for similarity of concept c with features transformed to embedding space of attribute that c most belongs to (learned)
        mask = torch.min(selected.unsqueeze(0), mask)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0)
        return mask[group]

    def relate_after(self, selected):
        if self.temporal_operator == 'linear':
            selected_padded = torch.zeros(self.max_num_segments, device=selected.device)
            selected_padded[:selected.size()[0]] = selected
            temporal_projection = self.after_projection_layer(selected_padded)
            return temporal_projection[:selected.size()[0]]
        elif self.temporal_operator == 'conv1d':
            return self.after_projection_layer(selected)
    
    def relate_before(self, selected):
        if self.temporal_operator == 'linear':
            selected_padded = torch.zeros(self.max_num_segments, device=selected.device)
            selected_padded[:selected.size()[0]] = selected
            temporal_projection = self.before_projection_layer(selected_padded)
            return temporal_projection[:selected.size()[0]]
        elif self.temporal_operator == 'conv1d':
            return self.before_projection_layer(selected)
    
    def unique(self, selected):
        if self.training or _test_quantize.value < InferenceQuantizationMethod.STANDARD.value:
            return jacf.general_softmax(selected, impl='standard', training=self.training)
        return jacf.general_softmax(selected, impl='standard', training=self.training)
    
    def intersect(self, selected1, selected2):
        return torch.min(selected1, selected2)
    
    def query(self, selected, group, attribute_groups):
        mask, word2idx = self._get_attribute_query_masks(attribute_groups)
        mask = (mask * selected.unsqueeze(-1).unsqueeze(0)).sum(dim=-2)
        if torch.is_tensor(group):
            return (mask * group.unsqueeze(1)).sum(dim=0), word2idx
        return mask[group], word2idx

    def _get_concept_groups_masks(self, concept_groups, k):
        if self._concept_groups_masks[k] is None:
            masks = list()
            for cg in concept_groups:
                if isinstance(cg, six.string_types):
                    cg = [cg]
                mask = None
                for c in cg:
                    new_mask = self.taxnomy[k].similarity(self.features[k], c)
                    mask = torch.min(mask, new_mask) if mask is not None else new_mask
                if k == 2 and _apply_self_mask['relate']:
                    mask = do_apply_self_mask(mask)
                masks.append(mask)
            self._concept_groups_masks[k] = torch.stack(masks, dim=0)
        return self._concept_groups_masks[k]
    
    def _get_attribute_query_masks(self, attribute_groups):
        if self._attribute_query_masks is None:
            masks, word2idx = list(), None
            for attribute in attribute_groups:
                mask, this_word2idx = self.taxnomy[1].query_attribute(self.features[1], attribute) # for each concept, calculates similarity to motions and how much concept belongs to attribute (shape num_segments x num_concepts)
                masks.append(mask)
                # sanity check.
                if word2idx is not None:
                    for k in word2idx:
                        assert word2idx[k] == this_word2idx[k]
                word2idx = this_word2idx

            self._attribute_query_masks = torch.stack(masks, dim=0), word2idx
        return self._attribute_query_masks

class DifferentiableReasoning(nn.Module):
    def __init__(self, used_concepts, input_dims, hidden_dims, temporal_operator, parameter_resolution='deterministic', vse_attribute_agnostic=False, max_num_segments=5):
        super().__init__()

        self.used_concepts = used_concepts
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.parameter_resolution = parameter_resolution
        
        self.temporal_operator = temporal_operator
        self.max_num_segments = max_num_segments

        if self.temporal_operator == 'linear': # linear
            self.before_projection_layer =  jacnn.LinearLayer(self.max_num_segments, self.max_num_segments, activation=None) # 5 max number of actions
            self.after_projection_layer =  jacnn.LinearLayer(self.max_num_segments, self.max_num_segments, activation=None) # 5 max number of actions
        elif self.temporal_operator == 'conv1d': # Conv1D
            print('using conv temporal operator')
            self.before_projection_layer = Conv1dTemporalProjection()
            self.after_projection_layer = Conv1dTemporalProjection()

        for i, nr_vars in enumerate(['attribute', 'relation']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars, concept_embedding.ConceptEmbedding(vse_attribute_agnostic))
            tax = getattr(self, 'embedding_' + nr_vars)
            rec = self.used_concepts[nr_vars]

            for a in rec['attributes']:
                tax.init_attribute(a, self.input_dims[1 + i], self.hidden_dims[1 + i])
            for (v, b) in rec['concepts']:
                tax.init_concept(v, self.hidden_dims[1 + i], known_belong=b)

        for i, nr_vars in enumerate(['attribute_ls', 'relation_ls']):
            if nr_vars not in self.used_concepts:
                continue
            setattr(self, 'embedding_' + nr_vars.replace('_ls', ''), concept_embedding_ls.ConceptEmbeddingLS(
                self.input_dims[1 + i], self.hidden_dims[1 + i], self.hidden_dims[1 + i]
            ))
            tax = getattr(self, 'embedding_' + nr_vars.replace('_ls', ''))
            rec = self.used_concepts[nr_vars]

            if rec['attributes'] is not None:
                tax.init_attributes(rec['attributes'], self.used_concepts['embeddings'])
            if rec['concepts'] is not None:
                tax.init_concepts(rec['concepts'], self.used_concepts['embeddings'])

    def forward(self, batch_features, progs):
        programs = []
        buffers = []
        result = []

        start_seg = 0
        for i, features in enumerate(batch_features):
            prog = progs[i]
            buffer = []

            buffer = []

            buffers.append(buffer)
            programs.append(prog)

            ctx = ProgramExecutorContext(self.embedding_attribute, self.embedding_relation, features, parameter_resolution=self.parameter_resolution, temporal_operator=self.temporal_operator, training=self.training, before_projection_layer=self.before_projection_layer, after_projection_layer=self.after_projection_layer, max_num_segments=self.max_num_segments)

            for block_id, block in enumerate(prog):
                op = block['op']

                if op == 'scene':
                    buffer.append(10 + torch.zeros(features[1].size(0), dtype=torch.float, device=features[1].device))
                    continue
                
                inputs = []
                
                for inp, inp_type in zip(block['inputs'], gdef.operation_signatures_dict[op][1]):
                    inp = buffer[inp]
                    if inp_type == 'object':
                        inp = ctx.unique(inp)
                    inputs.append(inp)
                
                if op == 'filter':
                    buffer.append(ctx.filter(*inputs, block['concept_idx'], block['concept_values']))
                elif op == 'relate':
                    if 'before' in block['relational_concept']:
                        buffer.append(ctx.relate_before(*inputs))
                    elif 'after' in block['relational_concept']:
                        buffer.append(ctx.relate_after(*inputs))
                    else:
                        assert(False)
                elif op == 'intersect':
                    buffer.append(ctx.intersect(*inputs))
                else:
                    assert block_id == len(prog) - 1, 'Unexpected query operation: {}. Are you using the CLEVR-convension?'.format(op)
                    if op == 'query':
                        buffer.append(ctx.query(*inputs, block['attribute_idx'], block['attribute_values']))


                if not self.training and _test_quantize.value > InferenceQuantizationMethod.STANDARD.value:
                    if block_id != len(prog) - 1:
                        buffer[-1] = -10 + 20 * (buffer[-1] > 0).float()
                
            result.append((op, buffer[-1]))

        return programs, buffers, result

class Conv1dTemporalProjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_intermediate_layers = 3

        self.layer1 = nn.Conv1d(1, 16, 3, padding=1) # in_channels, out_channels, kernel_size
        self.intermediate_layers = nn.ModuleList([nn.Conv1d(16, 16, 3, padding=(3-1)*(2**(i + 1)) // 2, dilation= 2**(i + 1)) for i in range(self.num_intermediate_layers)]) # padding = (kernel_size-1) * dilation // 2

        self.last_conv = nn.Conv1d(16, 1, 3, padding=1)
    
    def forward(self, x):
        x = self.layer1(x.unsqueeze(dim=0).unsqueeze(dim=0))
        x = nn.ReLU()(x)

        for layer in self.intermediate_layers:
            x = nn.ReLU()(layer(x)) + x # residual skip connections

        x = self.last_conv(x)
        x = x.squeeze()
        return x