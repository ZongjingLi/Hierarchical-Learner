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


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        '''
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        '''
        B, N, D = x.size()
        if self.residual:
            x = self.linear(x.view(B * N, D))
            x = self.relu(x + res.view(B * N, self.output_size))
        else:
            x = self.relu(self.linear(x.view(B * N, D)))

        return x.view(B, N, self.output_size)