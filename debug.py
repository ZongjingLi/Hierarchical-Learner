'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-03-22 17:00:05
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-03-22 17:00:26
 # @ Description: This file is distributed under the MIT license.
'''

'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:07:32
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:36:14
 # @ Description: This file is distributed under the MIT license.
'''


if __name__ == "__main__":
    from models import *
    from config import *

    input_ims = torch.randn([10,128,128,3])
    input_programs = ["Exist(Filter(Scene(), red))", "Count(Filter(Scene(), ship))"]

    inputs = {"images": input_ims, "programs": input_programs}

    hal_model = HierarchicalLearner(config)

    print("start the main function")
    #p = hal_model.executor.parse("exist(filter(scene(),red))")
    p = hal_model.parse("exist(scene())")
    p = hal_model.parse("unique(scene())")
    #p = hal_model.parse("exist(filter(scene(),red))")
    p = hal_model.executor.parse("filter(scene(),boat)")
    print(p)

    kwargs = {"features":torch.randn([8,200]),"end":torch.ones([8])}

    o = hal_model.executor(p,**kwargs)

    print(o["end"])

    optim = torch.optim.Adam(hal_model.parameters() , lr = 2e-2)
    for epoch in range(4000):
        o = hal_model.executor(p,**kwargs)
        loss = 0 - o["end"][0]  - o["end"][-4] 
        optim.zero_grad()
        loss.backward()
        optim.step()

    from datasets  import *
    from visualize import *
    hal_model = torch.load("checkpoints/joint_toy_slot_attention.ckpt", map_location = "cpu")
    hal_model.scene_perception = torch.load("checkpoints/toy_slot_attention.ckpt", map_location = "cpu")
    data = ToyDataWithQuestions("train")
    maps = [12, 33, 15, 25]

    ims = torch.cat([data[idx]["image"].unsqueeze(0) for idx in maps],0)
    outputs = hal_model.scene_perception(ims)
    visualize_outputs(ims,outputs)
    visualize_scores(outputs["object_scores"][:,:,0].detach().numpy())
    
    """
    inputs = {"image":ims,
            "question":[
                {
                    "program":[
                        "exist(filter(scene(),house))",
                        "exist(filter(scene(),house))",
                        "exist(filter(scene(),house))",
                        "exist(filter(scene(),house))"
                        ],
                    "answer":[0,1,2,3]
                }
                ]}

    model = torch.load("checkpoints/joint_toy_slot_attention.ckpt",map_location=config.device)
    outputs = model(inputs)
    visualize_outputs(inputs,outputs)
    
    #visualize_distribution(torch.sigmoid(o["end"]).detach().numpy())
    #print(outputs["object_scores"].shape)
    #visualize_distribution(outputs["object_scores"][0][...,0].detach().numpy())
    
    plt.show()
    """