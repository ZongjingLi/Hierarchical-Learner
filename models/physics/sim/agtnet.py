import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.linear0 = nn.Linear(input_dim,     hidden_dim)
        self.linear1 = nn.Linear(hidden_dim,    hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,    output_dim)
        self.activate = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, x):
        # B,N,D1 -> B,N,D2
        B,N,D = x.shape
        x = x.reshape(B * N, D)
        x = self.linear0(x)
        x = self.activate(x)
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        x = self.activate(x)
        x = x.reshape(B, N, D)
        return x

class ParticlePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.output_dim = output_dim
        self.linear0 = nn.Linear(input_dim,     hidden_dim)
        self.linear1 = nn.Linear(hidden_dim,    output_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # B,N,D1 -> B,N,D2
        B,N,D = x.shape 
        x = x.reshape([B * N, D])
        x = self.linear0(x)
        x = self.activate(x)
        x = self.linear1(x)
        return x.reshape(B,N,self.output_dim)

class RelationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.linear_0 = nn.Linear(input_dim,  hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        B ,N, Z = x.shape
        x = x.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape(B, N, self.output_dim)

class Propagator(nn.Module):
    def __init__(self, input_dim, output_dim, residual = False):
        super().__init__()
        self.residual = residual
        self.output_dim = output_dim
        self.linear_0 = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, res = None):
        B, N, D = inputs.shape
        if self.residual:
            x = self.linear_0(inputs.reshape(B * N, D))
            x = F.relu(x + res.reshape(B * N, self.output_dim))
        else:
            x = F.relu(self.linear_0([B * N, D]))
        return x.reshape(B, N, self.output_dim)

class PropModule(nn.Module):
    def __init__(self, config, input_dim, output_dim, batch = True, residual = False):
        super().__init__()
        device = config.device
        self.device = device
        self.config = config
        self.batch  = batch

        relation_dim = config.relation_dim

        effect_dim = pfd

        particle_feature_dim = config.particle_feature_dim
        relation_feature_dim = config.relation_feature_dim
        prop_feature_dim = config.prop_feature_dim
        pfd = prop_feature_dim

        self.residual = residual

        self.particle_encoder = ParticleEncoder(input_dim,pfd, particle_feature_dim)
        self.relation_encoder = RelationEncoder(2*input_dim + relation_dim,\
                                             relation_feature_dim, relation_feature_dim)
        self.particle_propagator = Propagator(2*pfd, pfd, self.residual)
        self.relation_propagator = Propagator(2*pfd, pfd)
        self.particle_predictor  = ParticlePredictor(pfd,output_dim,pfd)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, state, Rr, Rs, Ra, steps):
        # [Encode Particle Features]
        particle_effect = torch.autograd.Variable(\
            torch.zeros(state.shape[0],state.shape[1], self.effect_dim))
        particle_effect = particle_effect.to(self.device)

        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            print("Oh come on, why not use the batch-wise operation")
        # [Particle Encoder]
        particle_encode = self.particle_encoder(state)

        # [Relation Encoder] calculate the relation encoding
        relation_encode = self.relation_encoder(torch.cat([
            [state_r, state_s, Ra], 2
        ]))

        for i in range(steps):
            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else: pass

            # calculate relation effect

        pred = self.particle_predictor(particle_effect)

        return 

class AgtNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        position_dim = config.position_dim
        state_dim = config.state_dim
        action_dim = config.action_dim
        particle_fd = config.particle_feature_dim
        relation_fd = config.relation_feature_dim
        observation = config.observation

        self.particle_encoder = ParticleEncoder()
        self.particle_decoder = ParticlePredictor()

        self.propagator = Propagator()

    def forward(self, x):
        return x