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
    p = hal_model.executor.parse("exist(scene())")
    p = hal_model.executor.parse("unique(scene())")
    p = hal_model.executor.parse("exist(filter(scene(),red))")
    print(p)

    kwargs = {"features":torch.randn([3,200])}

    o = hal_model.executor(p,**kwargs)

    box_embeddings = hal_model.box_registry(torch.tensor([3,4]))
    c1 = box_embeddings[0:1,...]
    c2 = box_embeddings[1:2,...]
    bentailment = build_entailment(config)

    score = bentailment(c1,c2)

    print(o["end"])

    hal_model(inputs)

