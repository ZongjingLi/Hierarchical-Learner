'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:22:49
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:50
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def train(model,dataset,config):
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    dataloader = DataLoader(dataset, batch_size = config.batch_size)

    itrs = 0
    for epoch in range(config.epochs):
        for sample in dataloader:
            sample
            itrs += 1