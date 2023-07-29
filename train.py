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
from models import *
from visualize.answer_distribution import *
from visualize.visualize_pointcloud import *

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color

from legacy import *


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def log_imgs(imsize,pred_img,clusters,gt_img,writer,iter_):

    batch_size = pred_img.shape[0]
    
    # Write grid of output vs gt 
    grid = torchvision.utils.make_grid(
                          lin2img(torch.cat((pred_img.cpu(),gt_img.cpu()))),
                          normalize=True,nrow=batch_size)

    # Write grid of image clusters through layers
    cluster_imgs = []
    for i,(cluster,_) in enumerate(clusters):
        for cluster_j,_ in reversed(clusters[:i+1]): cluster = cluster[cluster_j]
        pix_2_cluster = to_dense_batch(cluster,clusters[0][1])[0]
        cluster_2_rgb = torch.tensor(color.label2rgb(
                    pix_2_cluster.detach().cpu().numpy().reshape(-1,imsize,imsize) 
                                    ))
        cluster_imgs.append(cluster_2_rgb)
    cluster_imgs = torch.cat(cluster_imgs)
    grid2=torchvision.utils.make_grid(cluster_imgs.permute(0,3,1,2),nrow=batch_size)
    writer.add_image("Clusters",grid2.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT",grid.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT Var",grid.detach().numpy(),iter_)

    visualize_image_grid(cluster_imgs[batch_size,...], row = 1, save_name = "val_cluster")
    visualize_image_grid(pred_img.reshape(batch_size,imsize,imsize,3)[0,...], row = 1, save_name = "val_recon")

def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]

def train_pointcloud(train_model, config, args, phase = "1"):
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = True if args.training_mode in ["joint", "query"] else False
    print("\nstart the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))

    #[setup the training and validation dataset]
    if args.dataset == "Objects3d":
        train_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))
    if args.dataset == "StructureNet":
        if phase in ["0","1"]:
            train_dataset = StructureDataset(config, category = "vase")
        if phase in ["2","3","4"]:
            train_dataset = StructureGroudingDataset(config)

    dataloader = DataLoader(train_dataset, batch_size = int(args.batch_size), shuffle = args.shuffle)

    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    if args.training_mode == "query":alpha = 0
    if args.training_mode == "perception":beta = 0
    

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    for epoch in range(args.epoch):
        epoch_loss = 0
        for sample in dataloader:
            sample, gt = sample
            # [perception module training]

            outputs = model.scene_perception(sample)
            all_losses = {}

            # [Perception Loss]
            perception_loss = 0
            for loss_name in outputs["loss"]:
                weight = args.loss_weights[loss_name]
                perception_loss += outputs["loss"][loss_name] * weight

            # [Language Loss]
            language_loss = 0
            if query:
                for question in sample["question"]:
                    for b in range(len(question["program"])):
                        program = question["program"][b] # string program
                        answer  = question["answer"][b]  # string answer

                        abstract_scene  = outputs["abstract_scene"]
                        top_level_scene = abstract_scene[-1]

                        working_scene = [top_level_scene]
                        
                        scores   = scores = outputs["abstract_scene"][-1]["scores"]
                        EPS = 1e-5
                        scores   = torch.clamp(scores, min = EPS, max = 1 - EPS).reshape([-1])
                        #scores = scores.unsqueeze(0)


                        features = top_level_scene["features"][b].reshape([scores.shape[0],-1])


                        edge = 1e-5
                        if config.concept_type == "box":
                            features = torch.cat([features,edge * torch.ones(features.shape)],-1)#.unsqueeze(0)

                        kwargs = {"features":features,
                                  "end":scores }

                        q = train_model.executor.parse(program)
                        
                        o = train_model.executor(q, **kwargs)
                        #print("Batch:{}".format(b),q,o["end"],answer)
                        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
                        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))

                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss += 0 #+F.mse_loss(int_num + 1,o["end"])
   
                            if itrs % args.checkpoint_itrs == 0:
                                #print(q,answer)
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).cpu().detach())
                                answer_distribution_num(o["end"].cpu().detach().numpy(),1+int_num.cpu().detach().numpy())
                        if answer in yes_or_no:
                            if answer == "yes":language_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
        
                            if itrs % args.checkpoint_itrs == 0:
                                #print(q,answer)
                                #print(torch.sigmoid(o["end"]).cpu().detach().numpy())
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).cpu().detach())
                                answer_distribution_binary(torch.sigmoid(o["end"]).cpu().detach().numpy())

            # [calculate the working loss]
            working_loss = perception_loss * alpha + language_loss * beta
            try:epoch_loss += working_loss.detach().numpy()
            except:epoch_loss += working_loss
            # [backprop and optimize parameters]
            for i,losses in enumerate(all_losses):
                for loss_name,loss in losses.items():
                    writer.add_scalar(str(i)+loss_name, loss, itrs)

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss, itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs):
                name = args.name
                expr = args.training_mode
                torch.save(train_model.state_dict(), "checkpoints/{}_{}_{}_{}_phase{}.pth".format(name,expr,config.domain,config.perception,phase))

                if args.dataset == "Objects3d":
                    point_cloud = sample["point_cloud"]
                    rgb = sample["rgb"]
                    coords = sample["coords"]
                    occ = sample["occ"]
                    coord_color = sample["coord_color"]
                    np.save("outputs/point_cloud.npy",np.array(point_cloud[0,:,:].cpu()))
                    np.save("outputs/rgb.npy",np.array(rgb[0,:,:].cpu()))
                    np.save("outputs/coords.npy",np.array(coords[0,:,:].cpu()))
                    np.save("outputs/occ.npy",np.array(occ[0,:,:].cpu()))
                    np.save("outputs/coord_color.npy",np.array(coord_color[0,:,:].cpu()))

                    recon_occ = outputs["occ"]
                    recon_coord_color = outputs["color"]
                    np.save("outputs/recon_occ.npy",np.array(recon_occ[0,:].cpu().unsqueeze(-1).detach()))
                    np.save("outputs/recon_coord_color.npy",np.array(recon_coord_color[0,:,:].cpu().detach()))
                if args.dataset == "StructureNet":
                    recon_pc = outputs["recon_pc"][0]
                    point_cloud = sample["point_cloud"][0]
                    masks = outputs["masks"][0]
                    np.save("outputs/recon_point_cloud.npy",np.array(recon_pc.cpu().detach()))
                    np.save("outputs/point_cloud.npy",np.array(point_cloud.cpu().detach()))
                    np.save("outputs/masks.npy",np.array(masks.cpu().detach()))
                    
            itrs += 1

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Percept:{} Language:{}, Time: {}".format(epoch + 1, itrs, working_loss,perception_loss,language_loss,datetime.timedelta(seconds=time.time() - start)))
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
    print("\n\nExperiment {} : Training Completed.".format(args.name))

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0,"chamfer":1.0,"equillibrium_loss":1.0}

argparser = argparse.ArgumentParser()
# [general config of the training]
argparser.add_argument("--phase",                   default = "0")
argparser.add_argument("--device",                  default = config.device)
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 400 * 3)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--batch_size",              default = 16)
argparser.add_argument("--dataset",                 default = "toy")
argparser.add_argument("--concept_type",            default = False)

# [perception and language grounding training]
argparser.add_argument("--perception",              default = "psgnet")
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 10.00)
argparser.add_argument("--beta",                    default = 0.001)
argparser.add_argument("--loss_weights",            default = weights)

# [additional training details]
argparser.add_argument("--warmup",                  default = True)
argparser.add_argument("--warmup_steps",            default = 300)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [curriculum training details]
argparser.add_argument("--effective_level",         default = 1)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 1000)
argparser.add_argument("--pretrain_perception",     default = False)

args = argparser.parse_args()

config.perception = args.perception
if args.concept_type: config.concept_type = args.concept_type

if args.checkpoint_dir:
    #model = torch.load(args.checkpoint_dir, map_location = config.device)
    model = SceneLearner(config)
    model.load_state_dict(torch.load(args.checkpoint_dir, map_location=args.device))
else:
    print("No checkpoint to load and creating a new model instance")
    model = SceneLearner(config)
model = model.to(args.device)


if args.pretrain_perception:
    model.scene_perception.load_state_dict(torch.load(args.pretrain_perception, map_location = config.device))

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            
            for r in range(1):
                random_long_range = torch.randint(128, (1,2) )[0]
                edges[0].append(random_long_range[0] // size)
                edges[1].append(random_long_range[1] % size)
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        if 1 and (i+dx) * size + (j + dy) != loc:
                            edges[0].append(loc)
                            edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

print("using perception: {} knowledge:{} dataset:{}".format(args.perception,config.concept_type,args.dataset))

if args.dataset in ["Objects3d","StructureNet"]:
    print("start the 3d point cloud model training.")
    train_pointcloud(model, config, args, phase = args.phase)

if args.dataset in ["Sprites","Acherus","Toys","PTR"]:
    print("start the image domain training session.")
    train_image(model, config, args)



