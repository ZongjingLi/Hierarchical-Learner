import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from config import *
from utils import *

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


root = config.dataset_root
cat = "chair"

pc_path = root + "/partnethiergeo/{}_geo/172.npz".format(cat)
pc_data = np.load(pc_path)

for name in pc_data.files:
    print(name,":", pc_data[name].shape)

hier_path = root + "/partnethiergeo/{}_hier/172.json".format(cat)
hier_data = load_json(hier_path)
for name in hier_data:
    print(name,":",0)

for component in hier_data["children"]:
    print(component)