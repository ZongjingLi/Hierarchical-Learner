from abc import abstractmethod
from collections import abc

import torch
from utils import *

class AbstractProgram:
    PROGRAM_REGISTRY = {}

    @property
    def name(self):return type(self).__name__

    @abstractmethod
    def __init__(self, *args):
        self.arguments = args