'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:31:26
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:10
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from utils import *



class ToyData(Dataset):
    def __init__(self,split = "train",resolution = (128,128)):
        super().__init__()
        
        assert split in ["train","val","test"]
        self.split = split
        self.resolution = resolution
        self.root_dir = "/Users/melkor/Documents/datasets/toy/images"

        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    def __len__(self): return 400#len(self.files)

    def __getitem__(self,index):
        image = Image.open(os.path.join(self.root_dir,"{}.png".format(index)))
        image = image.convert("RGB").resize(self.resolution) 
        image = self.img_transform(image).permute([1,2,0])
        sample = {"image":image}
        return sample