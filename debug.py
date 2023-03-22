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

    kwargs = {"features":torch.randn([8,200])}

    o = hal_model.executor(p,**kwargs)

    print(o["end"])

    hal_model(inputs)

    from datasets  import *
    from visualize import *
    data = ToyData("train")
    maps = [15,13,2,6]
    model = torch.load("checkpoints/toy_slot_attention.ckpt",map_location=config.device)
    inputs = torch.cat([data[idx]["image"].unsqueeze(0) for idx in maps],0)

    outputs = model(inputs)

    #visualize_outputs(inputs,outputs)
    #visualzie_distribution(torch.sigmoid(o["end"]).detach().numpy())
    plt.show()