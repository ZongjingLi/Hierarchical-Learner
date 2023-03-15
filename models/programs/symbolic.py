import re

import torch
import torch.nn.functional as F

from .abstract_program import AbstractProgram
from utils import copy_dict,apply,EPS

class SymbolicProgram(AbstractProgram):
    def __init__(self, *args):
        super().__init__(*args)
        self.kwargs = {}
        self.registered = None, []
