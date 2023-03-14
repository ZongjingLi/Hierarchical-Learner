'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:22:49
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:50
 # @ Description: This file is distributed under the MIT license.
'''

import os
import sys
import time
import datetime

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from datasets import *
from models.hal.model import HierarchicalLearner

def train(model,dataset,config):
    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = config.shuffle)

    itrs = 0
    start = time.time() # record the start time of the training
    for epoch in range(config.epochs):
        for sample in dataloader:
            itrs += 1

            # check to engage in warmup training 
            if itrs < config.warmup_step:
                learning_rate = config.lr * ((1 + itrs)/config.warmup_step)
            else:
                learning_rate = config.lr

            # check t engage in decay training

            sample
            working_loss = 0 # calculate the working loss of the batch

            optimizer.param_groups[0]["lr"] = learning_rate

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,
            datetime.timedelta(seconds=time.time() - start)))


from config import *
train_dataset = PartNet("train")
model = HierarchicalLearner(config)

train(model,train_dataset,config)