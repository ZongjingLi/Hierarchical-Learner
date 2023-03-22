'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:07:32
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:36:14
 # @ Description: This file is distributed under the MIT license.
'''


from models.nn.box_registry import build_box_registry


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
    p = hal_model.parse("exist(filter(scene(),red))")
    p = hal_model.parse("exist(filter(scene(),boat))")
    #p = hal_model.parse("count(scene())")
    print(p)

    kwargs = {"features":torch.randn([8,200])}

    o = hal_model.executor(p,**kwargs)

    print(o["end"])

    optim = torch.optim.Adam(hal_model.parameters(), lr = 2e-2)
    for epoch in range(10000):
        o = hal_model.executor(p,**kwargs)
        loss = 0 - F.logsigmoid(o["end"])
        loss.backward()
        optim.step()
        optim.zero_grad()
    
    print(o["end"])

    hal_model(inputs)
