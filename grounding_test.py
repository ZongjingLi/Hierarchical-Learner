import torch
import torch.nn as nn

from models import *
from config import *

config.perception = "csqnet"
learner = SceneLearner(config)

eps = 1e-6; d = 132
scores = [
    torch.tensor([1,0,0]).clamp(eps,1-eps),
    torch.tensor([0,0,0,1]).clamp(eps,1-eps)
    ]

features = [
    torch.randn([3,d]),
    torch.randn([4,d]),
    ]

connections = [
    torch.tensor([
        [1,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
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
for s in o["end"]:print((torch.sigmoid(s)+0.5).int())

q = learner.executor.parse("subtree(scene())")
print("subtrees",q)

o = learner.executor(q, **kwargs)
o["end"].reverse()
for s in o["end"]:print((torch.sigmoid(s)+0.5).int())

def visualize_tree(self, scores, connections, labels = None):
    pass