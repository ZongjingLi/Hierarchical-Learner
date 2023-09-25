import warnings
warnings.filterwarnings("ignore")

import tensorflow

import torch
import argparse 
import datetime
import time
import sys

from datasets import *

from config import *
from model import *
from visualize.answer_distribution import *
from visualize.visualize_pointcloud import *

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_hierarchy(model, depth):
    size = len(model.scene_builder)
    for i in range(1,size+1):
        if i <= depth:unfreeze_parameters(model.hierarhcy_maps[i-1])
        else:freeze_parameters(model.scene_builder)
    model.executor.effective_level = depth

def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]

def train(train_model, config, args, phase = "1"):
    B = int(args.batch_size)
    train_model = train_model.to(config.device)
    train_model.config = config
    train_model.executor.config = config
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = False if args.phase in ["0",0] else True
    clip_grads = query
    print("start the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    if args.phase in ["1",1]: args.loss_weights["equillibrium"] = 0.01
    #[setup the training and validation dataset]

    if args.dataset == "Objects3d":
        train_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))
    if args.dataset == "StructureNet":
        if args.phase in ["0",]:
            train_dataset = StructureDataset(config, category = "vase")
        if args.phase in ["1","2","3","4"]:
            train_dataset = StructureGroundingDataset(config, category = args.category, split = "train", phase = "1")
    if args.dataset == "Multistruct":
        train_dataset =multistructnet4096("train", "animal", False)
    
    #train_dataset = StructureGroundingDataset(config, category="vase", split = "train")
    dataloader = DataLoader(train_dataset, batch_size = int(args.batch_size), shuffle = args.shuffle)


    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    if args.training_mode == "query":alpha = 1
    if args.training_mode == "perception":beta = 1
    

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    if args.freeze_perception:
         print("freezed the perception module: True")
         freeze_parameters(train_model.part_perception)
    if phase not in ["0"]:
       freeze_hierarchy(train_model,int(phase))
         
    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./tf-logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)
    max_gradient = 1000.
    for epoch in range(args.epoch):
        epoch_loss =  0.0
        for sample in dataloader:
            sample, _ = sample
            percept_outputs = train_model.perception(sample)

            # [Percept Loss]
            percept_loss = 0.0
            losses = percept_outputs["losses"]
            for item in losses:
                percept_loss += losses[item]
                writer.add_scalar(item, losses[item].cpu().detach().numpy(), itrs)

            # [Query Loss]
            query_loss = 0.0
            
            # [Overall Working Loss]
            working_loss = percept_loss + query_loss
            writer.add_scalar("working_loss",working_loss.cpu().detach().numpy(), itrs)

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            if itrs % args.checkpoint_itrs == 0:
                torch.save()

            itrs += 1

        writer.add_scalar("epoch_loss", epoch_loss, epoch)
    print("\n\nExperiment {} : Training Completed.".format(args.name))




