import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class HierarchyBuilder(nn.Module):
    def __init__(self, config, num_slots):
        super().__init__()