import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class StructureDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        root = config.root
        path = root + "/partnethiergeo/"

    def __len__(self):return 0

    def __getitem__(self, idx):
        return idx