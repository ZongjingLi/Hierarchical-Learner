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
from datasets.ptr.ptr_datasets import PTRImageData
from datasets.sprites.sprite_dataset import SpriteWithQuestions
from models.hal.model import HierarchicalLearner
from models.percept.slot_attention import SlotAttentionParser, SlotAttentionParser64
from datasets import *
from visualize.answer_distribution import visualize_outputs,visualize_scores

def train(model,dataset,config,name):

    if config.domain == "ptr":
        dataset = PTRImageData("train")

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
                inputs = sample["image"].to(config.device)
                try:outputs = model.scene_perception(inputs)
                except:outputs = model(inputs)

                # get the components
                full_recon = outputs["full_recons"]
                recons     = outputs["recons"]
                masks      = outputs["masks"]
                loss       = outputs["loss"]
                if config.training_mode != "query":
                    working_loss += loss * 1000

            if config.training_mode == "query" or config.training_mode == "joint":
                query_loss = 0
                for question in sample["question"]:
                    for b in range(len(question["program"])):
                        program = question["program"][b] # string program
                        answer  = question["answer"][b]  # string answer

                        scores   = outputs["object_scores"][b,...,0] - EPS
                        features = outputs["object_features"][b]

                        edge = 1e-6
                        if config.concept_type == "box":
                            features = torch.cat([features,edge * torch.ones(features.shape, device = config.device)],-1)

                        kwargs = {"features":features,
                                  "end":scores }

                        q = model.executor.parse(program)
                        
                        o = model.executor(q, **kwargs)
                        #print("Batch:{}".format(b),q,o["end"],answer)
                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(config.device)
                            query_loss += F.mse_loss(int_num + 1,o["end"])
                        if answer in yes_or_no:
                            if answer == "yes":query_loss -= F.logsigmoid(o["end"])
                            else:query_loss -= torch.log(1 - torch.sigmoid(o["end"]))
                        #print(scores.float().detach().numpy())

                working_loss += query_loss * 0.003

             # calculate the working loss of the batch

            # back propagation to update the working loss
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            total_loss += working_loss.detach()

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))

            if itrs % config.ckpt_itr == 0:
                writer.add_scalar("working_loss", working_loss, itrs)
                if config.training_mode == "perception":
                    torch.save(model.scene_perception,"checkpoints/{}_{}_{}.ckpt".format(name,config.domain,config.perception))
                else:
                    torch.save(model,"checkpoints/{}_joint_{}_{}.ckpt".format(name,config.domain,config.perception))
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

                    visualize_outputs(inputs,outputs)
                    visualize_scores(outputs["object_scores"][:,:,0].cpu().detach().numpy())   

            itrs += 1
        total_loss = total_loss/len(dataloader)
        writer.add_scalar("epoch_loss",total_loss,epoch)



from config import *

# [Experiment Config]
exp_parser = argparse.ArgumentParser()
exp_parser.add_argument("--name",default = "Zilliax")
exp_parser.add_argument("--training_mode")
exp_parser.add_argument("--pretrain_perception",default = False)
exp_parser.add_argument("--pretrain_joint",default = False)
exp_parser.add_argument("--save_path",default = "checkpoints/")
experiment_config = exp_parser.parse_args()

print("Experiment Id: {} Mode: {}".format(experiment_config.name,experiment_config.training_mode))

# [Setup for the Perception Module Training]
if experiment_config.training_mode == "perception":
    model = HierarchicalLearner(config)
    train_dataset = ToyData("train")
    # adjust perception based on the pretrain model
    if experiment_config.pretrain_perception:
        config.training_mode == "perception"
        model.scene_perception = torch.load(experiment_config.pretrain_perception,map_location = config.device)
    else: model = HierarchicalLearner(config)

    # [Setup model device and more on objectness]
    model.executor.config = config.device
    model.scene_perception.ban_obj_score()

    # Put the model on the device
    model = model.to(config.device)
    train(model,train_dataset,config,experiment_config.name)

# [Setup for the Joint Training Case]
if experiment_config.training_mode == "joint":
    train_dataset = ToyDataWithQuestions("train")
    config.training_mode = "joint"
    config.warmup_steps = 500

    train_model = HierarchicalLearner(config)

    if experiment_config.pretrain_joint:
        train_model.scene_perception = torch.load(experiment_config.pretrain_joint,map_location = config.device)

    # [Setup model device and more on objectness]
    train_model.executor.config = config
    train_model.scene_perception.allow_obj_score()

    # Put the model on the device
    train_model = torch.load("checkpoints/KFT_joint_toy_slot_attention.ckpt", map_location = config.device)
    train_model = train_model.to(config.device)
    train_model.executor.config = config
    train(train_model,train_dataset,config, experiment_config.name)