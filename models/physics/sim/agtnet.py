import torch
import torch.nn as nn


class AgtNet(nn.Module):
    def __init__(self, config):
        super().__init__()