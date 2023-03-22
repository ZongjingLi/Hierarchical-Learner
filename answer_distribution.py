import numpy as np
import matplotlib.pyplot as plt

import torchvision

def visualize_outputs(image, outputs):

    full_recon = outputs["full_recons"]
    recons     = outputs["recons"]
    masks      = outputs["masks"]

    num_slots = recons.shape[1]
    
    # [Draw Components]
    plt.figure("Components")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    
    plt.imshow(recon_grid.permute(1,2,0).cpu().detach().numpy())

    # [Draw Masks]
    plt.figure("Masks")
    masks_grid = torchvision.utils.make_grid(masks.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(masks_grid.permute(1,2,0).cpu().detach().numpy())

    # [Draw Recons]
    plt.figure("Recons")
    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(recon_grid.permute(1,2,0).cpu().detach().numpy())

    # [Draw Full Recon]
    plt.figure("Full Recons",frameon=False)
    grid = torchvision.utils.make_grid(full_recon.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(grid.permute(1,2,0).cpu().detach().numpy())

    # [Draw GT Image]
    plt.figure("GT Image",frameon=False)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    gt_grid = torchvision.utils.make_grid(image.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=config.batch_size)
    plt.imshow(gt_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/gt_image.png")


if __name__ == "__main__":
    from datasets import *
    from config   import *
    data = ToyData("train")
    maps = [1,2,3,6]
    model = torch.load("checkpoints/toy_slot_attention.ckpt",map_location=config.device)
    inputs = torch.cat([data[idx]["image"].unsqueeze(0) for idx in maps],0)
    print(inputs.shape)
    outputs = model(inputs)
    visualize_outputs(inputs,outputs)
    plt.show()