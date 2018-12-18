import numpy as np
import torch
from torch import nn

from util import create_tens

class ActorNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, 
            hidden0=100, hidden1=50, final_w=0.003):
        super(ActorNetwork, self).__init__()
        # initialize network
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.norm0 = nn.BatchNorm1d(self.input_dim)

        self.func0 = nn.Linear(self.input_dim, hidden0)
        self.batch_norm0 = nn.BatchNorm1d(hidden0)

        self.func1 = nn.Linear(hidden0, hidden1)
        self.batch_norm1 = nn.BatchNorm1d(hidden1)

        self.func2 = nn.Linear(hidden1, output_dim)

        # initialize weights
        w0 = 1/ np.sqrt(self.input_dim)
        self.func0.weight.data = create_tens((hidden0, self.input_dim), w0)

        w1 = 1/np.sqrt(hidden0)
        self.func1.weight.data = create_tens((hidden1, hidden0), w1)

        self.func2.weight.data = create_tens((self.output_dim, hidden1), final_w) 

        # activation functions used
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward (self, x):
        z0 = self.ReLU(self.func0(x))
       
        z1 = self.ReLU(self.func1(z0))
       
        out = self.Tanh(self.func2(z1))
        return out




    
