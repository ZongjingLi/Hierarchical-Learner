
import torch
import torch.nn as nn

import numpy as np
from utils import *
from config import *

import matplotlib.pyplot as plt

root = config.dataset_root
cat = "chair"

path = root + "/partnethiergeo/{}_hier/172.json".format(cat)
data = load_json(path)
label = data["label"]
print("label",label)
print("edges",data["edges"])
print("box",data["box"])
print("children",data["children"])
print("id",data["id"])

def dfs(node):
    if "children" in node:
        for child in node["children"]:
            print(node["label"],"->")
            dfs(child)
    else:print(node["label"])

#dfs(data)
npt = 1000

def color(pc,pcc,node):
    if "children" in node:
        for child in node["children"]:
            lower = node["box"][:3]
            upper = node["box"][3:6]
            color(pcs, pcc, child)
    else:
        lower = torch.tensor(node["box"][:3]).unsqueeze(0).repeat(npt,1)
        upper = torch.tensor(node["box"][3:6]).unsqueeze(0).repeat(npt,1)
        print(node["box"])

        idx = torch.logical_and(pcc > lower , upper > pcc)
        print(idx.shape)
        
        pcc[idx] = 0.5

        print("render")


hier_path = root + "/partnethiergeo/{}_geo/172.npz".format(cat)
hier_data = np.load(hier_path)
print(hier_data.files)
for name in hier_data.files:
    print(name, hier_data[name].shape)



def visualize_pointcloud(input_pcs,name="pc"):
    rang = 0.4; N = len(input_pcs)
    fig = plt.figure("visualize",figsize=plt.figaspect(1/N), frameon = True)
    for i in range(N):
        ax = fig.add_subplot(1, N, 1 + i, projection='3d')
        ax.set_zlim(-rang,rang);ax.set_xlim(-rang,rang);ax.set_ylim(-rang,rang)
        # make the panes transparent
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_axis_off()
        ax.view_init(elev = -80, azim = -90)
        coords = input_pcs[i][0]
        colors = input_pcs[i][1]

        ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors)
    plt.savefig("outputs/{}.png".format(name))

n = 0
pcs = hier_data["parts"]
pcc = torch.clamp(torch.zeros([pcs[n].shape[0],3]) ** 2,0,1)

color(pcs,pcc, data)

visualize_pointcloud([(pcs[n],pcc )])
plt.show()