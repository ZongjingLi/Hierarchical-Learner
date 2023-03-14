if __name__ == "__main__":
    from models import *
    slot = SlotAttentionParser(8, 100, 3)
    inputs = torch.randn([10,128,128,4])
    outputs = slot(inputs)
    print("start the main function")