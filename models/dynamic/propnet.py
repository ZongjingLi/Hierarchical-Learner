import torch
import torch.nn as nn

class RelationalEncoder(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self,x):
        """
        input:  [B,N,N]
        output: [B,N,M]
        """
        B,N,D = x.shape
        return self.model(x.reshape([B * N, D])).reshape([B, N, D])

class ParticleEncoder(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self,x):
        """
        input:  [B,N,N]
        output: [B,N,M]
        """
        B,N,D = x.shape
        return self.model(x.reshape([B * N, D])).reshape([B, N, D])