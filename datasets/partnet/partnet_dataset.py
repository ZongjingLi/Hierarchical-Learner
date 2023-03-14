'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:31:26
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:10
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class PartNet(Dataset):
    def __init__(self,split = "train",data_path = None):
        super().__init__()
        
        assert(split in ["train","test","val"]),print("Unknown split for the dataset: {}.".format(split))
        
        self.split = split
        self.root_dir = ""

    def __getitem__(self,index):
        return index

    def __len__(self):return 6