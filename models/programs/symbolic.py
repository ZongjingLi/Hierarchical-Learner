import re

import torch
import torch.nn.functional as F

from .abstract_program import AbstractProgram
from utils import copy_dict,apply,EPS