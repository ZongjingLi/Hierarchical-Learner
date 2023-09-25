import torch
import torch.nn as nn

from .programs import *
from .percept import *
from .executor import *

class Halearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.perception = None

        self.executor = None
        
    
    def forward(self, sample):
        input_pc = sample["point_cloud"]