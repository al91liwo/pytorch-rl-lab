import numpy as np
import torch
from torch import nn

from util import create_tens


class CriticNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, 
            hidden0=400, hidden1=300, final_w = 0.003):
        
        super(CriticNetwork, self).__init__()
        # initialize network
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm0 = nn.BatchNorm1d(hidden0)
        
        self.func0 = nn.Linear(self.input_dim, hidden0)
        self.batch_norm0 = nn.BatchNorm1d(hidden1)
        
        self.func1 = nn.Linear(hidden0 + self.output_dim, hidden1)
        self.batch_norm1 = nn.BatchNorm1d(self.output_dim)

        self.func2 = nn.Linear(hidden1, self.output_dim)
        
        # initialize weights
        w0 = 1/ np.sqrt(self.input_dim)
        self.func0.weight.data = create_tens((hidden0, self.input_dim), w0)
        
        w1 = 1/np.sqrt(hidden0 + output_dim)
        self.func1.weight.data = create_tens((hidden1, hidden0 + self.output_dim), w1)
        
        self.func2.weight.data = create_tens((self.output_dim, hidden1), final_w)

        # used activation functions
        self.ReLU = nn.ReLU()

    def forward (self, x, y):
        x = self.func0(x)
        x = self.norm0(x)
        z0 = self.ReLU(x)

        # as in the paper actions were added in the second layer
        # paper: continuous control with deep reinforcement learning
        #print(z0, z0.shape, y, y.shape)
        z0 = torch.cat((z0, y), 1)
        z0 = self.func1(z0)
        z0 = self.batch_norm0(z0)
        
        z1 = self.ReLU(z0)
        z1 = self.func2(z1)
        z1 = self.batch_norm1(z1)

        out = self.ReLU(z1)
        return out
