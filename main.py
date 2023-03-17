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
    p = hal_model.parse("exist(union(scene(), scene()))")
    print(p)

    #q = p.evaluate(hal_model)

    hal_model(inputs)
    
