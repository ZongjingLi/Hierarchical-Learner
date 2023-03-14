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
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.hal.model import HierarchicalLearner
from datasets import *


def train(model,dataset,config):
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)

    # setup the checkpoint location and initalize the SummaryWritter
    writer = SummaryWriter(events_dir)

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = config.shuffle)
    
    checkpoint_path = config.ckpt_path
    if checkpoint_path:
        print("Loading model from %s" % checkpoint_path)
        model = torch.load(model,map_location=config.device)

    itrs = 0
    start = time.time() # record the start time of the training
    for epoch in range(config.epochs):
        for sample in dataloader:
            itrs += 1

            # check to engage in warmup training 
            if itrs < config.warmup_steps:
                learning_rate = config.lr * ((1 + itrs)/config.warmup_steps)
            else:
                learning_rate = config.lr
            # check to engage in decay training
            if itrs > config.decay_steps:
                learning_rate = config.lr * (config.decay_rate ** (itrs / config.decay_steps))
            else:
                learning_rate = config.lr
            optimizer.param_groups[0]["lr"] = learning_rate # setup the lr params

            # execute the model according to the training mode
            sample
            working_loss = 100 * 2**(0-itrs/100) # calculate the working loss of the batch

            # back propagation to update the working loss
            

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,
            datetime.timedelta(seconds=time.time() - start)))

            if itrs % config.ckpt_itr == 0:
                #grid = torchvision.utils.make_grid(0,normalize=True,nrow=config.batch_size)
                writer.add_scalar("working_loss", working_loss, itrs)
                torch.save(model,"checkpoints/test.ckpt")


from config import *
train_dataset = PartNet("train")
model = HierarchicalLearner(config)

train(model,train_dataset,config)