import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils import load_json
import networkx as nx

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

class StructureGroundingDataset(Dataset):
    def __init__(self, config,category = "vase" , split = "train", phase = "1"):
        super().__init__()
        root = config.dataset_root
        self.root = root
        self.split = split
        if isinstance(category, str): cats = [category]
        else: cats = category
        self.valid_types = ["geometry"]
        self.data_idx = []
        for cat in cats:
            with open(root + "/partnethiergeo/{}_geo/{}.txt".format(cat,split),"r") as split_idx:
                split_idx = split_idx.readlines()
                for i in range(len(split_idx)):
                    self.data_idx.append([cat, split_idx[i].strip()])
    
    def __len__(self):return len(self.data_idx)

    def __getitem__(self, idx):
        category,index = self.data_idx[idx] 
        root = self.root 
        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/{}.npy".format(category,self.split,index)
        point_cloud = np.load(pc_path)

        qa_file = load_json(root + "/partnet_geo_qa/{}/{}/qa/{}.json".format(category,self.split,index))["all"]
        questions   =  []
        answers     =  []
        programs    =  []
        for item in qa_file:
            if item["type"] in self.valid_types:
                questions.append(item["question"])
                answers.append(item["answer"])
                programs.append(item["program"])
        scene_tree = root + "/partnet_geo_qa/{}/{}/annotations/{}.pickle".format(category,self.split,index)
        #nx.read_gpickle(\
            

        return {"point_cloud":point_cloud,"questions":questions,"answers":answers,\
            "scene_tree":scene_tree,"programs":programs,"index":index,"category":category}, 0
