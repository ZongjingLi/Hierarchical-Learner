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
from bleach import Cleaner

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.sprites.sprite_dataset import SpriteWithQuestions
from models.hal.model import HierarchicalLearner
from models.percept.slot_attention import SlotAttentionParser, SlotAttentionParser64
from datasets import *

def train(model,dataset,config):

    if config.training_mode == "joint":
        try:model.perception.allow_obj_score()
        except:pass

    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)

    # setup the checkpoint location and initalize the SummaryWritter
    writer = SummaryWriter(events_dir)

    if config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    if config.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = config.lr)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle = config.shuffle)
    
    checkpoint_path = config.ckpt_path
    if checkpoint_path:
        print("Loading model from %s" % checkpoint_path)
        model = torch.load(model,map_location=config.device)

    itrs = 0
    start = time.time() # record the start time of the training
    for epoch in range(config.epochs):
        total_loss = 0
        for sample in dataloader:

            # check to engage in warmup training 
            if config.warmup and itrs < config.warmup_steps:
                learning_rate = config.lr * ((1 + itrs)/config.warmup_steps)
            else:
                learning_rate = config.lr
            # check to engage in decay training
            if config.decay and itrs > config.decay_steps:
                learning_rate = config.lr * (config.decay_rate ** (itrs/config.decay_steps))

            optimizer.param_groups[0]["lr"] = learning_rate # setup the lr params

            working_loss = 0
            # execute the model according to the training mode
            if True or config.training_mode == "perception" or config.training_mode == "joint":
                inputs = sample["image"]
                try:outputs = model.perception(inputs)
                except:outputs = model(inputs)

                # get the components
                full_recon = outputs["full_recons"]
                recons     = outputs["recons"]
                masks      = outputs["masks"]
                loss       = outputs["loss"]
                if config.training_mode != "query":
                    working_loss += loss 

            if config.training_mode == "query" or config.training_mode == "joint":
                query_loss = 0
                for question in sample["question"]:
                    for b in range(len(question["program"])):
                        program = question["program"][b] # string program
                        answer  = question["answer"][b]  # string answer

                        scores   = outputs["object_scores"][b,...,0] - EPS
                        features = outputs["object_features"][b]
                        
                        edge = 1e-4
                        features = torch.cat([features,edge * torch.ones(features.shape)],-1)

                        kwargs = {"features":features,
                                  "end":scores }

                        q = model.executor.parse(program)
                        
                        o = model.executor(q, **kwargs)

                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float()
                            query_loss += F.mse_loss(int_num,o["end"])
                        if answer in yes_or_no:
                            if answer == "yes":query_loss -= F.logsigmoid(o["end"])
                            else:query_loss -= F.logsigmoid(1 - o["end"])
                        
                        #torch.log(torch.sigmoid(o["end"]))

                working_loss += query_loss

             # calculate the working loss of the batch

            # back propagation to update the working loss
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            total_loss += working_loss.detach()

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))

            if itrs % config.ckpt_itr == 0:
                writer.add_scalar("working_loss", working_loss, itrs)
                torch.save(model,"checkpoints/joint_{}_{}.ckpt".format(config.domain,config.perception))
                if config.training_mode == "joint" or config.training_mode == "query":
                    writer.add_scalar("qa_loss", query_loss, itrs)
                writer.add_scalar("vision_loss", loss, itrs)
                if True or config.training_mode == "perception" or config.training_mode == "joint":
                    # load the images, reconstructions, and other thing
                    num_slots = recons.shape[1]

                    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
                    writer.add_image("Recons",recon_grid.cpu().detach().numpy(),itrs)
    
                    masks_grid = torchvision.utils.make_grid(masks.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
                    writer.add_image("Masks",masks_grid.cpu().detach().numpy(),itrs)

                    comps_grid = torchvision.utils.make_grid((recons*masks).cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
                    writer.add_image("Components",comps_grid.cpu().detach().numpy(),itrs)

                    grid = torchvision.utils.make_grid(full_recon.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
                    writer.add_image("Full Recons",grid.cpu().detach().numpy(),itrs)

                    gt_grid = torchvision.utils.make_grid(sample["image"].cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
                    writer.add_image("GT Image",gt_grid.cpu().detach().numpy(),itrs)

                    writer.add_image("Backup",gt_grid.cpu().detach().numpy(),itrs)             

            itrs += 1
        total_loss = total_loss/len(dataloader)
        writer.add_scalar("epoch_loss",total_loss,epoch)

from config import *

train_dataset = ToyData("train")
train_dataset = ToyDataWithQuestions("train")

model = HierarchicalLearner(config)
slotmodel = torch.load("checkpoints/toy_slot_attention.ckpt",map_location=config.device)
model.perception = slotmodel

config.training_mode = "joint"
config.warmup_steps = 500
model.perception.allow_obj_score()

train(model,train_dataset,config)