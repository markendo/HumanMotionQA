import torch.nn as nn
import jactorch.nn as jacnn

from jacinle.logging import get_logger
from jacinle.utils.container import G
from datasets.definition import gdef



logger = get_logger(__file__)

__all__ = ['make_motion_reasoning_configs', 'MotionReasoningModel']


class Config(G):
    pass

def make_base_configs():
    configs = Config()

    configs.data = G()
    configs.model = G()
    configs.train = G()
    configs.train.weight_decay = 1e-4

    return configs

def make_motion_reasoning_configs():
    configs = make_base_configs()
    configs.model.sg_dims = [None, 256, 256]
    configs.model.vse_known_belong = False
    configs.model.vse_large_scale = False
    configs.model.vse_hidden_dims = [None, 64, 64]
    return configs


class MotionReasoningModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        from models.agcn import Model as AGCNModel
        self.motion_encoder = AGCNModel(False)

        import models.quasi_symbolic as qs
        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(configs.model.vse_large_scale, configs.model.vse_known_belong),
            configs.model.sg_dims, configs.model.vse_hidden_dims, self.temporal_operator, max_num_segments=self.max_num_segments
        )

        import models.losses as vqalosses
        self.query_loss = vqalosses.QueryLoss()
        self.filter_monitor = vqalosses.IntermediateFilterMonitor()

    def train(self, mode=True):
        super().train(mode)

    def _make_vse_concepts(self, large_scale, known_belong):
        return {
            'attribute': {
                'attributes': list(gdef.attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(gdef.relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in gdef.relational_concepts.items() for v in vs
                ]
            }
        }