
import torch
from config import *
from train import *

import matplotlib.pyplot as plt

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0,"chamfer":100.0,"equillibrium_loss":1.0}

argparser = argparse.ArgumentParser()
# [general config of the training]
argparser.add_argument("--phase",                   default = "0")
argparser.add_argument("--device",                  default = config.device)
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 400 * 3)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--batch_size",              default = 1)
argparser.add_argument("--dataset",                 default = "StructureNet")
argparser.add_argument("--category",                default = ["vase"])
argparser.add_argument("--freeze_perception",       default = False)
argparser.add_argument("--concept_type",            default = False)

# [perception and language grounding training]
argparser.add_argument("--perception",              default = "csqnet")
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 1.00)
argparser.add_argument("--beta",                    default = 1.0)
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
argparser.add_argument("--checkpoint_itrs",         default = 10,       type=int)
argparser.add_argument("--pretrain_perception",     default = False)

args = argparser.parse_args()

if __name__ == "__main__":
    args.checkpoint_dir = "checkpoints/KFT_StructureNet_0.ckpt"
    args.checkpoint_dir = None

model = Halearner(config)

if args.checkpoint_dir:
    if args.checkpoint_dir[-4:] == "ckpt":model = torch.load(args.checkpoint_dir, map_location=args.device)

print("start the demo")

B = 1
if args.dataset == "StructureNet":
    dataset = StructureGroundingDataset(config, category="vase", split = "train")
if args.dataset == "Multistruct":dataset = multistructnet4096("train","airplane", False)
dataloader = DataLoader(dataset, batch_size = B, shuffle = True)

# [Get A Sample Data]
for sample in dataloader:
    sample, gt = sample

model.perception.split_components = True
outputs = model.perception(sample)

components = outputs["components"]

coords = outputs["recon_pc"].detach()
n = coords.shape[0]
coords_colors = torch.ones([n,3]) * 0.5

in_coords = sample["point_cloud"][0]
n = in_coords.shape[0]
in_colors = torch.ones([n,3]) * 0.5
input_pcs = [(coords,coords_colors),[in_coords, in_colors]]

visualize_pointcloud(input_pcs)
plt.show()

def visualize_pointcloud_components(pts):
    namo_colors = [
    '#1f77b4',  # muted blue
    '#005073',  # safety orange
    '#96aab3',  # cooked asparagus green
    '#2e3e45',  # brick red
    '#08455e',  # muted purple
    '#575959',  # chestnut brown
    '#38677a',  # raspberry yogurt pink
    '#187b96',  # middle gray
    '#31393b',  # curry yellow-green
    '#1cd1ed'   # blue-teal
    ]
    all_pts = pts.reshape([-1,3])
    N = pts.shape[0]
    rang = 0.7
    s = 3.0
    fig = plt.figure("visualize",figsize=(s * N//2,s * 2), frameon = True)
    full_fig = plt.figure("full_viz", figsize = (s,s), frameon = False)
    full_rows = 2
    full_cols = N // 2 
    full_ax = full_fig.add_subplot(1,1,1,projection='3d')
    full_ax.set_axis_off()
    full_ax.set_zlim(-rang,rang);full_ax.set_xlim(-rang,rang);full_ax.set_ylim(-rang,rang)
    for i in range(N):
        row = i // full_cols
 
        ax = fig.add_subplot(2, full_cols , 1 + i  , projection='3d')
        ax.set_zlim(-rang,rang);ax.set_xlim(-rang,rang);ax.set_ylim(-rang,rang)
        # make the panes transparent
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            #axis._axinfo['grid']['linewidth'] = 0.5
            #axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_axis_off()
        #ax.view_init(elev = -80, azim = -90)
        for j in range(N):
            coords = pts[j]
            colors = namo_colors[j]
            alpha = 1.0 if i == j else 0.05
            ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors, alpha = alpha)
        coords = pts[i]
        colors = namo_colors[i]
        full_ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors)
    fig.savefig("outputs/pc_components.png")
    full_fig.savefig("outputs/pc_full_components.png")

visualize_pointcloud_components(components[0].detach())
plt.show()