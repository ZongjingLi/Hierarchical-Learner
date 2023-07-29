from config     import *
from models     import *
from datasets   import *

# [Create a Dataset]
B = 1
dataset = StructureGroundingDataset(config, category="vase", split = "train")
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

def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]


learner = SceneLearner(config)
optimizer = torch.optim.RMSprop(learner.parameters(), lr = 1e-4)

print(sample["programs"])

print(sample["questions"])

print(sample["answers"])

for epoch in range(10000):
    g = -0.5; r = 0.3
    outputs = learner.scene_perception(sample)


    features,masks,positions = outputs["features"],outputs["masks"],outputs["positions"] 


    features = learner.feature2concept(features)
    features = torch.cat([features, torch.ones([1,features.shape[1],config.concept_dim]) * EPS], dim = -1)

    qa_programs = sample["programs"]
    answers = sample["answers"]
    scene = learner.build_scene(features)
    for b in range(B):
        scores,features,connections = load_scene(scene, b)

        kwargs = {"features":features,
                  "end":scores,
                 "connections":connections}

        language_loss = 0
        for i,q in enumerate(qa_programs):
            q = learner.executor.parse("subtree(scene())")

            o = learner.executor(q, **kwargs)
            o["end"].reverse()

            #for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))

            #q = learner.executor.parse("exist(filter(subtree(scene()),chair ))")
            q = learner.executor.parse("exist(subtree(scene()) )")

            o = learner.executor(q, **kwargs)
            #print(o["end"])
            language_loss += (1 + torch.sigmoid( (o["end"] + g) )/r)
        
        optimizer.zero_grad()
        language_loss.backward()
        optimizer.step()

        
        o = learner.executor(q, **kwargs)
        sys.stdout.write("\r p:{}".format(float(torch.sigmoid( (o["end"] + g)/r ).detach()) ) )