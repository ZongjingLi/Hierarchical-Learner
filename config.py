import torch
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",default = device)
config = parser.parse_args(args = [])