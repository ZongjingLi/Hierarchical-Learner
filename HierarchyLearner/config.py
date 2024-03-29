import torch
import argparse
from model import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count,
              "parents":Parents,"subtree":Subtree}

LOCAL = True

root_path = "/Users/melkor/Documents/GitHub/Hierarchical-Learner/HierarchyLearner"\
     if LOCAL else "Hierarcical-Learner/HierarchyLearner"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--root",               default = root_path)
parser.add_argument("--dataset_root",       default = "/Users/melkor/Documents/datasets")
parser.add_argument("--device",             default = device)
parser.add_argument("--name",               default = "HierarchyLearner")

parser.add_argument("--domain",             default = "structure")
parser.add_argument("--category",           default = ["chair"])

parser.add_argument("--perception",         default = "point_net")
parser.add_argument("--scaling",            default = 1.0)

# acne network
parser.add_argument("--latent_dim",         default = 128)
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

parser.add_argument("--object_num",         default = 11)
parser.add_argument("--part_num",           default = 3)
parser.add_argument("--hierarchy_latent",   default = 128)
parser.add_argument("--hierarchy_construct",default = [4,1])

# setup the concept learner 
parser.add_argument("--concept_projection", default = True)
parser.add_argument("--concept_type",       default = "cone")
parser.add_argument("--concept_dim",        default = 100)
parser.add_argument("--object_dim",         default = 100)
parser.add_argument("--temperature",        default = 0.2)

# box concept methods
parser.add_argument("--method",             default = "uniform")
parser.add_argument("--offset",             default = [-.25, .25])
parser.add_argument("--center",             default =[.0, .0])
parser.add_argument("--entries",            default = 32 * 3)
parser.add_argument("--translator",         default = translator)

config = parser.parse_args(args = [])