import torch
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",             default = device)

# hiearchical model setup
parser.add_argument("--perception",         default = "slot_attention")
parser.add_argument("--concept_type",       default = "box")

# training setup 
parser.add_argument("--epochs",             default = 100)
parser.add_argument("--batch_size",         default = 2)
parser.add_argument("--optimizer",          default = "Adam")
parser.add_argument("--lr",                 default = 2e-4)
parser.add_argument("--warmup",             default = True)
parser.add_argument("--decay",              default = True)

# save checkpoints setup
parser.add_argument("--ckpt_itr",           default = 100)
config = parser.parse_args(args = [])