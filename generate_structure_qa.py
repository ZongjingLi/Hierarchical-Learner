import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from visualize.visualize_pointcloud import *
from config import *
from utils import *
import networkx as nx
import os
import random

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

colors = [torch.tensor(hex2rgb(color))/255.0 for color in colors]
color_names = ["red" for _ in range(10)]

def find_outliers(pc):
    # Calculate mean distance from each point to every other point
    pc_expand = pc[None, :, :].repeat(pc.shape[0], axis=0)
    pc_expand = pc_expand - pc[:, None, :].repeat(pc.shape[0], axis=1)
    mean_dists = np.mean(np.linalg.norm(pc_expand, axis=2), axis=1)
    # Assume mean distances are normally distributed
    mean_dists = (mean_dists - np.mean(mean_dists)) / np.std(mean_dists)
    return mean_dists > 3.0

def unit_normalize_pc(pc, filter_outliers=True,):
    if filter_outliers:
        # Calculate outliers
        mask = find_outliers(pc)
        # Filter out the really extreme points
        first_ind = np.argwhere(~mask)[0][0]
        pc[mask] = pc[first_ind]
    # Normalized point cloud to unit sphere
    # Center the point cloud at origin

    mean_coord = pc.mean(0)
    pc = pc - mean_coord
    # Calculate the distance between the origin and the furthest point
    dists = np.linalg.norm(pc, axis=1)
    max_dist = np.max(dists)
    # Divide points by max_dist and fit inside unit sphere
    return pc / max_dist

def calculate_dists_between_pointsets(points, new_points):
    # points is a matrix of B x N x C
    # new_points is a matrix of B x M x C
    # Returns matrix B x M x N of distances from points
    dists = -2 * torch.bmm(new_points, points.mT)
    dists += torch.sum(torch.pow(points, 2), 2)[:, None, :]
    dists += torch.sum(torch.pow(new_points, 2), 2)[:, :, None]
    return dists


def fpsampling(pointset, m, return_cent_inds=False, random_start=True):
    # Iterative farthest point sampling
    # The next point added is the point that is the farthest away from the nearest centroid
    # pointset is a matrix of B x N x 3
    # Returns matrix of B x M x 3 centroid points
    # Returns matrix B x M x N of distances from points
    num_batch, num_points, _ = pointset.size()
    batch_idx = torch.arange(num_batch, dtype=torch.long).to(pointset.device)
    # Initialize centroid points
    centroids = torch.zeros(num_batch, m, dtype=torch.long).to(pointset.device)
    actual_cents = torch.zeros(num_batch, m, 3).to(pointset.device)
    # Initialize distances
    distances = torch.ones(num_batch, num_points).to(pointset.device) * 1e10
    if random_start:
        farthest = torch.randint(0, num_points, (num_batch,),
                                 dtype=torch.long).to(pointset.device)
    else:
        farthest = torch.zeros(num_batch, dtype=torch.long).to(pointset.device)
    # Iterative point sampling
    for i in range(m):
        # Select next centroids
        centroids[:, i] = farthest
        actual_cents[:, i, :] = pointset[batch_idx, farthest, :]
        # Calculate distances and next point
        dists = torch.square(pointset - actual_cents[:, i:i+1, :]).sum(dim=2)
        distances = torch.min(distances, dists)
        farthest = torch.max(distances, 1)[1]
    if return_cent_inds:
        return centroids, actual_cents
    return actual_cents

def create_box(box_params):
    center = box_params[0: 3]
    lengths = box_params[3: 6]
    dir_1 = box_params[6: 9]
    dir_2 = box_params[9:]
    #
    s = 1.0
    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    #
    d1 = s * 0.5*lengths[0]*dir_1
    d2 = s * 0.5*lengths[1]*dir_2
    d3 = s * 0.5*lengths[2]*dir_3
    #
    verts = np.zeros([8, 3])
    verts[0, :] = center - d1 - d2 - d3
    verts[1, :] = center - d1 + d2 - d3
    verts[2, :] = center + d1 - d2 - d3
    verts[3, :] = center + d1 + d2 - d3
    verts[4, :] = center - d1 - d2 + d3
    verts[5, :] = center - d1 + d2 + d3
    verts[6, :] = center + d1 - d2 + d3
    verts[7, :] = center + d1 + d2 + d3
    #
    faces = np.zeros([6, 4], dtype='int64')
    faces[0, :] = [3, 2, 0, 1]
    faces[1, :] = [4, 6, 7, 5]
    faces[2, :] = [0, 2, 6, 4]
    faces[3, :] = [3, 1, 5, 7]
    faces[4, :] = [2, 3, 7, 6]
    faces[5, :] = [1, 0, 4, 5]
    #
    return verts, faces


root = config.dataset_root

def generate(cat = "chair", idx = 172, pts_num = 2048):
    pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(cat, idx)
    pc_data = np.load(pc_path)

    # [Point Cloud]
    pc = torch.tensor(pc_data["parts"][0])

    hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(cat, idx)
    hier_data = load_json(hier_path)

    pc, rgbs = dfs_point_cloud(pc_data["parts"], get_leafs(hier_data))
    valid_indices = random.sample(list(range(pc.shape[0])), pts_num)
    pc = pc[valid_indices,:]
    rgbs = rgbs[valid_indices,:]
    
    scene_tree = get_scene_tree(hier_data)
 
    return {"point_cloud":pc,"rgbs":rgbs}

def get_scene_tree(h):
    return 

    
def get_leafs(h):
    nodes = []
    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_leafs(child))
        return nodes
    else:
        return [h["id"]]

def get_labels(h):
    nodes = []
    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_labels(child))
        return nodes
    else:
        return [h["label"]]

def build_scene_tree(h):
    nodes = []

    if "children" in h:
        for child in h["children"]:
            nodes.extend(get_labels(child))
        return nodes
    else:
        return [h["label"]]

def dfs_point_cloud(pc_data, nodes = None):
    point_cloud = []
    rgbs = []
    num_pt = 1000
    fig = plt.figure()
    ax = Axes3D(fig)
    if nodes is None: nodes = range(pc_data.shape[0])
    for i in nodes:
        k = np.random.randint(0,9)
        color_name = color_names[k]
        color = torch.tensor(colors[k])
        pts = torch.tensor(pc_data[i])
        ax.scatter(pts[:,0],pts[:,1],pts[:,2], color = [np.array(color) for _ in range(pts.shape[0])])    
        point_cloud.append(pts)
        rgbs.append(color.unsqueeze(0).repeat(num_pt,1))
    point_cloud = torch.cat(point_cloud, dim = 0)
    rgbs = torch.cat(rgbs, dim = 0)
    return point_cloud, rgbs

def generate_structure(cat = "chair", idx = 176):
    pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(cat, idx)
    pc_data = np.load(pc_path)

    # [Point Cloud]
    pc = torch.tensor(pc_data["parts"][0])

    # [Build Scene Tree]
    hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(cat, idx)
    hier_data = load_json(hier_path)

    scene_tree = nx.Graph()
    def build(h,root):
        name = h["label"]
        scene_tree.add_node(name)
        scene_tree.add_edge(root, name)
        if "children" in h:
            for child in h["children"]:
                build(child, name)
    build(hier_data, "root")

    return {"point_cloud":pc, "rgbs":None,"scene_tree":scene_tree}

#outputs = generate(idx = 176)

outputs = generate_structure()
st = outputs["scene_tree"]
nx.draw_networkx(st)
plt.show()

plt.show()

if outputs["rgbs"] is None: outputs["rgbs"] = .5 * torch.ones([outputs["point_cloud"].shape[0],3] )

visualize_pointcloud([
    (outputs["point_cloud"], outputs["rgbs"])
])
plt.show()