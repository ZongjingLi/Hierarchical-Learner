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

def train(train_model, config, args, phase = "0", num_sample = None):
    B = int(args.batch_size)
    train_model = train_model.to(config.device)
    train_model.config = config
    train_model.executor.config = config
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = False if phase in ["0",0] else True
    clip_grads = query
    print("start the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    if args.phase in ["1",1]: args.loss_weights["equillibrium"] = 0.01
    #[setup the training and validation dataset]

    if args.dataset == "Objects3d":
        train_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))
    if args.dataset == "StructureNet":
        if phase in ["0",]:
            train_dataset = StructureDataset(config, category = args.category[0])
        if phase in ["1","2","3","4"]:
            train_dataset = StructureGroundingDataset(config, category = args.category[0], split = "train", phase = "1")
    if args.dataset == "Multistruct":
        train_dataset =multistructnet4096("train", "animal", False)
    
    if num_sample is not None: train_dataset = torch.utils.data.Subset(train_dataset, list(range(num_sample)))

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
    if True or num_sample is None:
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
            if query:
                features = percept_outputs["features"]

                #for k in sample:print(k)
                qa_programs = sample["programs"]
                answers = sample["answers"]

                features = train_model.feature2concept(features)
                if config.concept_type == "box" and False:

                    features = torch.cat([
                        features, 
                        EPS * torch.ones(B,features.shape[1],config.concept_dim)\
                        ],dim = -1)
                #print(features.shape)
                scene = train_model.build_scene(features)

            
                for b in range(features.shape[0]):
                    scores,features,connections = load_scene(scene, b)

                    kwargs = {"features":features,
                    "end":scores,
                    "connections":connections}

                    for i,q in enumerate(qa_programs):
                        answer = answers[i][b]

                        q = train_model.executor.parse(q[0])

                        o = train_model.executor(q, **kwargs)
                        
                        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
                        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))
                        
                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            query_loss += F.mse_loss(int_num ,o["end"])
                            
                        if answer in yes_or_no:
                            if answer == "yes":
                                query_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:
                                query_loss -= torch.log(1 - torch.sigmoid(o["end"]))
            
            # [Overall Working Loss]
            working_loss = percept_loss * alpha + query_loss * beta
            epoch_loss += working_loss.cpu().detach().numpy()

            writer.add_scalar("working_loss",working_loss.cpu().detach().numpy(), itrs)

            # [Optimization Loss]
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            if itrs % args.checkpoint_itrs == 0:
                torch.save(train_model, "checkpoints/{}_{}_{}.ckpt".format(args.name,args.dataset,args.phase))

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Percept:{} Language:{}, Time: {}"\
                .format(epoch + 1, itrs, working_loss,percept_loss,query_loss,datetime.timedelta(seconds=time.time() - start)))
            itrs += 1

        writer.add_scalar("epoch_loss", epoch_loss, epoch)
    print("\n\nExperiment {} : Training Completed.".format(args.name))
    return train_model



