import torch
import torch.nn as nn

from models import *
from config import *

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import networkx as nx

config.perception = "csqnet"
learner = SceneLearner(config)

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

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


"""
pot
body
container
containing_things
liquid_or_soil
plant
other
lid
base
foot_base
foot
"""

gt_tree = nx.DiGraph()
gt_tree.add_nodes_from(["pot","body","container","plant"])
gt_tree.add_edges_from([
    ("pot","body"),
    ("body","container"),
    ("body","plant")
])


colors = [(1.0,1.0,1.0,1.0) for _ in range(4)]

plt.figure("gt_tree")
plt.subplot(1,2,1)
layout = nx.layout.planar_layout(gt_tree.nodes)

nx.draw_networkx_nodes(gt_tree,layout,label = nx.nodes)
nx.draw_networkx_edges(
    gt_tree,pos=layout,alpha=(0.4,0.3,0.1),width=5)
#plt.subplot(1,2,1)
#nx.draw_networkx(gt_tree,nx.layout.shell_layout(gt_tree.nodes), node_color = colors)

eval_data = []
for node in gt_tree.nodes:
    eval_data.append({"program":"","answer":"yes"})
for edge in gt_tree.edges:
    eval_data.append({"program":"","answer":"yes"})

def build_label(feature, executor):
    default_label = "x"
    default_color = [0,0,0,0.1]
    predicates = executor.concept_vocab
    prob = 0.1
    for predicate in predicates:
        logit = executor.entailment(
            feature,executor.get_concept_embedding(predicate)).unsqueeze(-1) 
        if logit > 0:
            default_label = predicate
            prob = torch.sigmoid((logit - 0.5)/0.5).detach()
    default_color = [1,0,0.4,float(max(prob,0.1))]
    return default_label, default_color

def visualize_outputs(scores, features, connections,executor):
    shapes = [score.shape[0] for score in scores]
    nodes = []
    labels = []
    colors = []
    edges = []
    edge_alphas = []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            nodes.append(sum(shapes[:i]) + j)
            l,c = build_label(features[i][j], executor)
            c[0] = [0.2,0.4,0.6][i]
            labels.append(l)
            colors.append(c)

    for n in range(len(connections)):
        connection = connections[n]
        r,c = connection.shape
        for i in range(r):
            for j in range(c):
                u = i + sum(shapes[:n])
                v = j + sum(shapes[:n+1])
                edges.append((v,u))
                #print(u,v,float(connection[i][j].detach()))
                edge_alphas.append(float(connection[i][j].detach()))

    csq_tree = nx.DiGraph()
    csq_tree.add_nodes_from(nodes)
    csq_tree.add_edges_from(edges)
    layout = nx.layout.kamada_kawai_layout(csq_tree)
  
    nx.draw_networkx_nodes(csq_tree, layout, node_color = colors)

    nx.draw_networkx_edges(
    csq_tree,pos=layout,alpha=edge_alphas,width=1)


model = SceneLearner(config)
optim = torch.optim.Adam(model.parameters(), lr = 2e-4)

struct = config.hierarchy_construct
node_features = [
    torch.randn([1,struct[i],100]) for i in range(3)
]

scores = [
    torch.ones(1),
    torch.ones(3),
    torch.ones(4),
]
scores.reverse()

features = [
    torch.randn(1),
    torch.randn(3),
    torch.randn(4),
]
features.reverse()

connections = [
    torch.tensor([
        [1.0],
        [1.0],
        [0.3]
    ]),
    torch.tensor([
        [0.8,0.0,0.2],
        [0.0,0.1,0.9],
        [0.1,0.9,0.0],
        [0.0,0.9,0.2],
    ]),

]
connections.reverse()

plt.subplot(122)
visualize_outputs(scores, features, connections, model.executor)
plt.show()

for epoch in range(1000):
    
    loss = 0
    for qa_pair in eval_data:
        q = qa_pair[""]
        loss += 0
    optim.zero_grad()
    loss.backward()
    optim.step()

#for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))