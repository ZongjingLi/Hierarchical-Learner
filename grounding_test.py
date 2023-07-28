import torch
import torch.nn as nn

from models import *
from config import *

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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

colors = np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in colors])
tree_color = colors[-1]

def visualize_tree(scores,connections,labels = None, sigma = False):

    fig, ax = plt.subplots()
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    min_v = 0.1; max_v = 1.0
    if labels is None: labels = [
        ["p" for _ in range(len(scores[t]))] for t in range(len(scores))
    ]
    layouts = []; eps = 0.00; width = 0.4; n = 1

    # Draw Nodes and Possible Patches of the Tree
    for i,idx in enumerate(range(len(scores))):
        score = scores[i]; ss = len(score)
        layout = torch.linspace(-(width)**i,(width)**i,ss)
        alphas = torch.sigmoid(scores[i]).cpu().detach() \
            if sigma else scores[i].cpu().detach()
        alphas = alphas.clamp(min_v,max_v)
        ax.scatter(layout,torch.tensor(i/n).repeat(ss), \
            alpha = alphas, c = [tree_color for _ in range(ss)],linewidths=12)
        for u in range(ss): 
            plt.text(layout[u]-0.02,i/n,labels[i][u])
            im = plt.imread('assets/ice.png')
            oi = OffsetImage(im, zoom = 0.15, alpha = float(alphas[u]))
            box = AnnotationBbox(oi, (layout[u]-0.02,i/n), frameon=False)
            ax.add_artist(box)
        layouts.append([layout,i/n])

    # Draw All the Connections between Nodes
    for k in range(len(connections)):
        connection = connections[k]
        lower_layout = layouts[k]
        upper_layout = layouts[k+1]
        for i in range(len(lower_layout[0])):
            for j in range(len(upper_layout[0])):
                alpha = float(connection[j][i])
                ax.plot(
                    (lower_layout[0][i],upper_layout[0][j]),
                    (lower_layout[1] + eps,upper_layout[1] - eps),
                    color = "gray", alpha = alpha
                    ) 
    ax.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
eps = 1e-6; d = 132
scores = [
    torch.tensor([1,0.0,0.9]).clamp(eps,1-eps),
    torch.tensor([0,0,0,0,0,0.9]).clamp(eps,1-eps)
    ]

features = [
    torch.randn([3,d]),
    torch.randn([5,d]),
    ]

connections = [
    torch.tensor([
        [1,1,0,0,0,0],
        [0,0,1,0,0,1],
        [0,0,0,1,1,0],
    ]).float()
]

for term in [scores, features, connections]:term.reverse()

kwargs = {"features":features,
          "end":scores,
          "connections":connections}

q = learner.executor.parse("parents(scene())")
print("parents:",q)

o = learner.executor(q, **kwargs)
o["end"].reverse()
for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))

q = learner.executor.parse("subtree(scene())")
print("subtree",q)

o = learner.executor(q, **kwargs)
o["end"].reverse()
for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))

#visualize_tree(scores,connections)
#plt.show()

# there is an $p1$ contains a $p2$
# exist(filter(subtree(filter(scene(),p1)),p2))

import matplotlib.pyplot as plt
import networkx as nx

G = nx.balanced_tree(3, 5)
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=20, alpha=0.5, node_color="blue", with_labels=False)
plt.axis("equal")
plt.show()