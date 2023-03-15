'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 09:33:13
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:36:32
 # @ Description: This file is distributed under the MIT license.
'''


import torch
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",             default = device)

# part-centric perception module setup
parser.add_argument("--imsize",             default = 128)
parser.add_argument("--perception",         default = "slot_attention")
parser.add_argument("--num_slots",          default = 5)
parser.add_argument("--object_dim",         default = 100)
parser.add_argument("--slot_itrs",          default = 3)

# hiearchical concept model setup
parser.add_argument("--concept_type",       default = "box")
parser.add_argument("--concept_dim",        default = 100)

# training setup 
parser.add_argument("--training_mode",      default = "perception")
parser.add_argument("--epochs",             default = 1000)
parser.add_argument("--batch_size",         default = 2)
parser.add_argument("--optimizer",          default = "Adam")
parser.add_argument("--lr",                 default = 2e-4)
parser.add_argument("--warmup",             default = True)
parser.add_argument("--warmup_steps",       default = 1000)
parser.add_argument("--decay",              default = False)
parser.add_argument("--decay_steps",        default = 2000)
parser.add_argument("--decay_rate",         default = 0.99)
parser.add_argument("--shuffle",            default = True)

# save checkpoints setup
parser.add_argument("--ckpt_itr",           default = 50)
parser.add_argument("--ckpt_path",          default = False)

config = parser.parse_args(args = [])