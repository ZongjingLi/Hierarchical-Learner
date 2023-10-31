import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from karanir.utils import load_json
import networkx as nx

class StructureDataset(Dataset):
    def __init__(self, config, category = "chair", mode = "train", split = "train"):
        super().__init__()
        root = config.dataset_root
        self.root = root
        self.category = category
        cats = [category]
        self.split = "train"

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

        # [Load Coords and Others]
        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/coord_{}.npy".format(category,self.split,index)
        query_coords = torch.tensor(np.load(pc_path)).float()

        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/color_{}.npy".format(category,self.split,index)
        query_colors = torch.tensor(np.load(pc_path)).float()

        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/occ_{}.npy".format(category,self.split,index)
        query_occ = torch.tensor(np.load(pc_path)).float()

        return {"point_cloud":torch.tensor(pc_data["parts"][0]).float(),
                "rgb":torch.ones([pc_data["parts"][0].shape[0],1]),\
                "coords":query_coords,
                "coord_color":query_colors,
                "occ":query_occ},\
                pc_data["parts"][0]

class StructureGroundingDataset(Dataset):
    def __init__(self, config,category = "chair" , split = "train", phase = "1"):
        super().__init__()
        root = config.dataset_root
        self.root = root
        self.split = split
        if isinstance(category, str): cats = [category]
        else: cats = category
        print("StructureGrouding Net with ",category)
        self.valid_types = ["existence","hierarchy"]
        if phase in [2,"2"]: self.valid_types.append("counting")
        self.data_idx = []
        for cat in cats:
            with open(root + "/partnethiergeo/{}_geo/{}.txt".format(cat,split),"r") as split_idx:
                split_idx = split_idx.readlines()
                for i in range(len(split_idx)):
                    self.data_idx.append([cat, split_idx[i].strip()])
        self.phase = phase
        self.qa_size = 12
    
    def __len__(self):return len(self.data_idx)

    def __getitem__(self, idx):
        category,index = self.data_idx[idx] 
        root = self.root 
        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/{}.npy".format(category,self.split,index)
        point_cloud = torch.tensor(np.load(pc_path)).float()
        
        # [Load Coords and Others]
        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/coord_{}.npy".format(category,self.split,index)
        query_coords = torch.tensor(np.load(pc_path)).float()

        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/color_{}.npy".format(category,self.split,index)
        query_colors = torch.tensor(np.load(pc_path)).float()

        pc_path = root + "/partnet_geo_qa/{}/{}/point_cloud/occ_{}.npy".format(category,self.split,index)
        query_occ = torch.tensor(np.load(pc_path)).float()


        phase = self.phase
        qa_file = load_json(root + "/partnet_geo_qa/{}/{}/qa/{}.json".format(category,self.split,index))["all"][phase]
        questions   =  []
        answers     =  []
        programs    =  []
        for item in qa_file:
            if item["type"] in self.valid_types:
                questions.append(item["question"])
                answers.append(item["answer"])
                programs.append(item[list(item.keys())[0]])
        scene_tree = root + "/partnet_geo_qa/{}/{}/annotations/{}.pickle".format(category,self.split,index)

        idxs = [np.random.choice(list(range(len(questions)))) for _ in range(self.qa_size)]

        r_questions = [questions[i] for i in idxs]
        r_answers   = [answers[i] for i in idxs]
        r_programs  = [programs[i] for i in idxs]
        
        #{'point_cloud': point_cloud.float(),
        #        'rgb': rgb_world.float(),
        #        'coords': coord.float(),
        #        'intrinsics': intrinsics.float(),
        #        'cam_poses': np.zeros(1),
        #        'occ': torch.from_numpy(labels).float(),
        #        'coord_color': coord_color.float(),
        #        'id': index
        #        } 

        return {"point_cloud":point_cloud,"questions":r_questions,"answers":r_answers,\
            "scene_tree":scene_tree,"programs":r_programs,"index":index,"category":category\
            ,"coords":query_coords,"coord_color":query_colors,"occ":query_occ,"rgb":torch.ones([point_cloud.shape[0],1])}, 0
