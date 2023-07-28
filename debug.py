from config     import *
from models     import *
from datasets   import *

# [Create a Dataset]
B = 1
dataset = StructureDataset(config)
dataloader = DataLoader(dataset, batch_size = B)

# [Get A Sample Data]
for sample in dataloader:
    sample, gt = sample

EPS = 1e-5
config.perception = "csqnet"
config.training_mode = "3d_perception"
config.concept_type = "box"
config.concept_dim = 100 
config.domain = "structure"



learner = SceneLearner(config)
optimizer = torch.optim.Adam(learner.parameters(), lr = 2e-1)

outputs = learner.scene_perception(sample)

features,masks,positions = outputs["features"],outputs["masks"],outputs["positions"] 


features = learner.feature2concept(features)
features = torch.cat([features, torch.ones([1,10,config.concept_dim]) * EPS], dim = -1)

def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]

scene = learner.build_scene(features)
for b in range(B):
    scores,features,connections = load_scene(scene, b)

    kwargs = {"features":features,
          "end":scores,
          "connections":connections}

    q = learner.executor.parse("subtree(scene())")

    o = learner.executor(q, **kwargs)
    o["end"].reverse()

    for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))

    q = learner.executor.parse("exist(filter(subtree(scene()), chair))")

    o = learner.executor(q, **kwargs)
    print(o["end"])

    optimizer.zero_grad()
    o["end"].backward()
    optimizer.step()

    q = learner.executor.parse("exist(subtree(scene()))")

    o = learner.executor(q, **kwargs)
    print(o["end"])