'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 17:32:52
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 17:34:49
 # @ Description: This file is distributed under the MIT license.
'''


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