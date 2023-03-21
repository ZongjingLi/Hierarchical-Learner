import torch
import torch.nn as nn

from models.nn import *
from models.nn.box_registry import build_box_registry
from models.percept import *
from .executor import *
from utils import *

class UnknownArgument(Exception):
    def __init__():super()

class HierarchicalLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        # [Unsupervised Part-Centric Representation]
        if config.perception == "slot_attention":
            if config.imsize == 128:self.scene_perception = SlotAttentionParser(config.num_slots,config.object_dim,config.slot_itrs)
            else:self.scene_perception = SlotAttentionParser64(config.num_slots,config.object_dim,config.slot_itrs)
        else:
            raise UnknownArgument

        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)

        # [Neuro Symbolic Executor]
        self.executor = HalProgramExecutor(config)

    def parse(self, program):return self.executor.parse(program)
    
    def forward(self, inputs):

        # [Parse the Input Scenes]
        part_centric_output = self.scene_perception(inputs["images"])

        scene_parsed = part_centric_output

        programs = inputs["programs"]

        outputs = {}
        return outputs