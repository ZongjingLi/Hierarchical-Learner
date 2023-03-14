import torch
import torch.nn as nn

def train(model,dataset,config):
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)