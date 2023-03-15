import torch
from torch import nn

class BoxRegistry(nn.Module):
    _init_methods = {"uniform": torch.nn.init.uniform_}

    def __init__(self, config):
        super().__init__()
        self.dim = config.concept_dim

        registry_config = config
        entries = config.entries