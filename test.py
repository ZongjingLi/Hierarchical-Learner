import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x, flag = False):
        if flag:
            return self.k * x + self.b
        else:
            return self.k * x

model = LinearModel()

inputs = torch.linspace(-1,1,10)

gt = inputs * 5 + 3

outputs = model(inputs, flag = False)

optim = torch.optim.Adam(model.parameters(), lr = 1e-1)

epochs = 400
for epoch in range(epochs):
    outputs = model(inputs, True)
    
    # calculate loss
    loss = F.mse_loss(outputs, gt)
    
    optim.zero_grad()
    loss.backward()
    optim.step()

print(model.k,model.b)
