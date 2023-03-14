import torch
import torch.nn as nn

from models.nn import *

class HierarchicalLearner(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.proj = FCBlock(3, 128, 3, 3)

    def forward(self,inputs):
        outputs = {}
        return outputs