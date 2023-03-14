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
from datasets.spirtes.sprite_dataset import SpriteWithQuestions
from models.hal.model import HierarchicalLearner
from models.percept.slot_attention import SlotAttentionParser, SlotAttentionParser64
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

            # check to engage in warmup training 
            if config.warmup and itrs < config.warmup_steps:
                learning_rate = config.lr * ((1 + itrs)/config.warmup_steps)
            else:
                learning_rate = config.lr
            # check to engage in decay training

            optimizer.param_groups[0]["lr"] = learning_rate # setup the lr params

            working_loss = 0
            # execute the model according to the training mode
            if config.training_mode == "perception":
                inputs = sample["image"]
                outputs = model(inputs)

                # get the components
                full_recon = outputs["full_recons"]
                recons     = outputs["recons"]
                masks      = outputs["masks"]
                loss       = outputs["loss"]
                working_loss += loss

            if config.training_mode == "query":
                pass
            if config.training_mode == "joint":
                pass
             # calculate the working loss of the batch

            # back propagation to update the working loss
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))

            if itrs % config.ckpt_itr == 0:
                writer.add_scalar("working_loss", working_loss, itrs)
                torch.save(model,"checkpoints/test.ckpt")

                if config.training_mode == "perception":
                    # load the images, reconstructions, and other thing
                    grid = torchvision.utils.make_grid(full_recon.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
                    writer.add_image("Full Recons",grid.cpu().detach().numpy(),itrs)

                    gt_grid = torchvision.utils.make_grid(sample["image"].cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
                    writer.add_image("GT Image",gt_grid.cpu().detach().numpy(),itrs)

                    num_slots = recons.shape[1]

                    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
                    writer.add_image("Recons",recon_grid.cpu().detach().numpy(),itrs)
    
                    masks_grid = torchvision.utils.make_grid(masks.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
                    writer.add_image("Masks",masks_grid.cpu().detach().numpy(),itrs)

            itrs += 1

from config import *
train_dataset = PartNet("train")
train_dataset = SpriteWithQuestions("train",resolution = (config.imsize,config.imsize))
model = HierarchicalLearner(config)
model = SlotAttentionParser64(5,100,6)

train(model,train_dataset,config)