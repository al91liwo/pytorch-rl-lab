import numpy as np
from torch import nn
import torch

from util import create_tens

class ActorNetwork(nn.Module):
    
    def __init__(self, layers, activations=None, final_w=0.003):
        super(ActorNetwork, self).__init__()
        if activations == None:
            self.activations = [nn.ReLU6()]*(len(layers)-1)
        else:
            self.activations = activations

        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])

        # initialize weights
        for i in range(len(self.layers)-2):
            w = 1/ np.sqrt(layers[i])
            self.layers[i].weight.data = create_tens((layers[i+1], layers[i]), w)

        self.layers[-1].weight.data = create_tens((layers[-1], layers[-2]), final_w)

    def forward(self, x):
        output = x
        for i in range(len(self.layers)-1):
            output = self.activations[i](self.layers[i](output))
        output = self.layers[-1](output)
        return output




    
