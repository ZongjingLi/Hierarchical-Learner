import torch
import argparse
from model import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count,
              "parents":Parents,"subtree":Subtree}

LOCAL = True

root_path = "/Users/melkor/Documents/GitHub/HierarchyLearner" if LOCAL else ""

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--root",               default = root_path)
parser.add_argument("--dataset_root",       default = "/Users/melkor/Documents/datasets")
parser.add_argument("--device",             default = device)
parser.add_argument("--name",               default = "HierarchyLearner")

#parser.add_argument("--domain",             default = "ptr")
#parser.add_argument("--category",           default = "")

parser.add_argument("--domain",             default = "structure")
parser.add_argument("--category",           default = ["vase"])

# acne network
parser.add_argument("--num_pts",            default = 1024)
parser.add_argument("--indim",              default = 3)
parser.add_argument("--grid_dim",           default = 256)
parser.add_argument("--decoder_grid",       default = "learnable")
parser.add_argument("--decoder_bottleneck_size",    default = 1280) # 1280
parser.add_argument("--acne_net_depth",     default = 3)
parser.add_argument("--acne_num_g",         default = 10)
parser.add_argument("--acne_dim",           default = 128)
parser.add_argument("--acne_bn_type",       default = "bn")
parser.add_argument("--cn_type", type=str,
                      default="acn_b",
                      help="Encoder context normalization type")
parser.add_argument("--node_feat_dim",      default = 102)
parser.add_argument("--pose_code",          default = "nl-noR_T")

config = parser.parse_args(args = [])