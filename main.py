'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:07:32
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:36:14
 # @ Description: This file is distributed under the MIT license.
'''


if __name__ == "__main__":
    from models import *
    slot = SlotAttentionParser(8, 100, 3)
    inputs = torch.randn([10,128,128,3])
    outputs = slot(inputs)
    print("start the main function")