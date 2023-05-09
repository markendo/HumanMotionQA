from jacinle.logging import get_logger
from datasets.definition import DatasetDefinitionBase
import jacinle.io as io
from datasets.humanmotionqa.utils import program_to_nsclseq

logger = get_logger(__file__)

__all__ = [
    'HumanMotionQADefinition',
]

class HumanMotionQADefinition(DatasetDefinitionBase):
    operation_signatures = [
        ('scene', [], [], 'object_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'], ['object'], 'object_set'),
        ('query', ['attribute'], ['object'], 'word'),
        ('intersect', [], ['object_set', 'object_set'], 'object_set'),
    ]

    relational_concepts = {
        'temporal_relation': ['before', 'after']
    }

    attribute_concepts = {'action': ['walk', 'place something', 'take/pick something up', 'knock', 'throw', 'catch', 'stand up', 'sit', 'run', 'squat', 'eat', 'jump', 'kneel', 'crouch', 'kick', 'cartwheel', 'skip', 'jumping jacks', 'jump rope', 'jog', 'dribble', 'leap', 'limp', 'punch', 'tie', 'juggle', 'clap', 'wave', 'bow', 'crawl', 'fall', 'salute', 'shuffle', 'drink', 'flip', 'duck', 'lunge', 'march', 'flail arms', 'yawn', 'golf', 'mix'], 'direction': ['forward', 'backwards', 'right', 'left'], 'body_part': ['right hand', 'right arm', 'left hand', 'left leg', 'right foot', 'right leg', 'left arm', 'left foot'], 'interaction_object': ['ball', 'door', 'cup', 'phone', 'shelf']}

    def program_to_nsclseq(self, program, question=None):
        return program_to_nsclseq(program)

    def update_collate_guide(self, collate_guide):
        # Scene annotations.
        for attr_name in self.attribute_concepts:
            collate_guide['attribute_' + attr_name] = 'concat'
            collate_guide['attribute_relation_' + attr_name] = 'concat'
        for attr_name in self.relational_concepts:
            collate_guide['relation_' + attr_name] = 'concat'

        # From ExtractConceptsAndAttributes and SearchCandidatePrograms.
        for param_type in self.parameter_types:
            collate_guide['question_' + param_type + 's'] = 'skip'
        collate_guide['program_parserv1_groundtruth_qstree'] = 'skip'
        collate_guide['program_parserv1_candidates_qstree'] = 'skip'
        


