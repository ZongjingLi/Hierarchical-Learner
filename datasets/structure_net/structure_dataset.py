import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils import load_json

class StructureDataset(Dataset):
    def __init__(self, config, category = "chair", mode = "train"):
        super().__init__()
        root = config.dataset_root
        self.root = root
        self.category = category
        cats = [category]

        #hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(category)
        self.data_idx = []
        for cat in cats:
            with open(root + "/partnethiergeo/{}_geo/{}.txt".format(cat,mode),"r") as split_idx:
                split_idx = split_idx.readlines()
                for i in range(len(split_idx)):
                    self.data_idx.append([cat, split_idx[i].strip()])

    def __len__(self):return len(self.data_idx)

    def __getitem__(self, idx):
        category,index = self.data_idx[idx] 
        root = self.root 
        pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(category,index)
        pc_data = np.load(pc_path)
        return {"point_cloud":pc_data["parts"][0]}, pc_data["parts"][0]

class StructureGroudingDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        root = config.root
        path = root + "/partnetheirgeo"
    
    def __len__(self):return 0

    def __getitem__(self, idx):
        return idx