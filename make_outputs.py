from visualize.visualize_pointcloud import *
import numpy as np
import torch
import argparse

visparser = argparse.ArgumentParser()
visparser.add_argument("--model",           default = "point_net")
visconfig = visparser.parse_args()

if visconfig.model in ["csq_net"]:
    coords = np.load("outputs/recon_point_cloud.npy")
    n = coords.shape[0]
    colors = torch.ones([n,3]) * 0.5

    pc = np.load("outputs/point_cloud.npy")
    pc_colors = torch.ones(pc.shape[0],3) * 0.5

    input_pcs = [(coords,colors),(pc,pc_colors)]

    visualize_pointcloud(input_pcs)
    plt.show()

if visconfig.model in ["point_net"]:
    point_cloud = np.load("outputs/point_cloud.npy")
    rgb = np.load("outputs/rgb.npy")
    coords = np.load("outputs/coords.npy")
    occ = np.load("outputs/occ.npy")
    coord_color = np.load("outputs/coord_color.npy")

    recon_occ = np.load("outputs/recon_occ.npy")
    recon_coord_color = np.load("outputs/recon_coord_color.npy")

    recon_probs = recon_occ
    #recon_probs = (occ+1)/2

    input_pcs = [(coords * (occ+1)/ 2,coord_color),
             (point_cloud,rgb),
             (coords * recon_probs,recon_coord_color)]

    visualize_pointcloud(input_pcs)
    plt.show()