
from models import *
from config import *

model = AutoLearner(config)

print(model)

B = 2
T = 32
N = 11
D = 2

states = torch.randn([B,T,N,D])
Rr = torch.ones([N,N,3])
Rs = torch.ones([N,N,2])
Ra = torch.ones([N,N,2])

data = states, Rr, Rs, Ra