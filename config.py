import torch
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",         default = device)

# training setup 
parser.add_argument("--epochs",         default = 100)
parser.add_argument("--optimizer",      default = "Adam")
parser.add_argument("--lr",             default = 2e-4)
config = parser.parse_args(args = [])