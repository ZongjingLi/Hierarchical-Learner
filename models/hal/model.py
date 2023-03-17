import torch
import torch.nn as nn

from models.nn import *
from models.percept import *
from .executor import *

class UnknownArgument(Exception):
    def __init__():super()

class HierarchicalLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        if config.perception == "slot_attention":
            if config.imsize == 128:self.scene_perception = SlotAttentionParser(config.num_slots,config.object_dim,config.slot_itrs)
            else:self.scene_perception = SlotAttentionParser64(config.num_slots,config.object_dim,config.slot_itrs)
        else:
            raise UnknownArgument
        self.proj = FCBlock(3, 128, 3, 3)

        self.executor = HalProgramExecutor(config)

    @staticmethod
    def parse(self, string, translator = None):
        if translator == None: translator = self.config.translator
        def chain(p):
            head = p[]
            
            pass
        program = chain(string)
        return program
    
    def forward(self,inputs):
        part_centric_output = self.scene_perception(inputs["images"])

        scene_parsed = part_centric_output

        programs = inputs["programs"]

        outputs = {}
        return outputs