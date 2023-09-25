import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

class DynamicSprites(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]