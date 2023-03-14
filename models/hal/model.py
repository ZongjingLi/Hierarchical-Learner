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
            if config.imsize == 128:self.scene_perception = SlotAttentionParser(config.num_slots,config.object_dim,config.slot_itrs)
            else:self.scene_perception = SlotAttentionParser64(config.num_slots,config.object_dim,config.slot_itrs)
        else:
            raise UnknownArgument
        self.proj = FCBlock(3, 128, 3, 3)


    def forward(self,inputs):
        outputs = {}
        return outputs