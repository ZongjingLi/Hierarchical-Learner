import torch
import torch.nn as nn

from models.nn import *
from models.percept import *

class UnknownArgument(Exception):
    def __init__():super()

class HierarchicalLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        if config.perception == "slot_attention":
            self.scene_perception = SlotAttentionParser
        else:
            raise UnknownArgument
        self.proj = FCBlock(3, 128, 3, 3)


    def forward(self,inputs):
        outputs = {}
        return outputs