import numpy as np
from torch import nn

from src.util import create_tens


class ClampTanh(nn.Module):
    """
    ClampTanh is a customizable tanh activation function.
    It is a tanh function that is sized to output values from low to high.
    """
    def __init__(self, low, high):
        super(ClampTanh, self).__init__()
        self.tanh = nn.Tanh()
        if low >= high:
            print("low should be smaller than high!")
        self.range = (high - low)/2
        self.center = (low + high)/2

    def forward(self, x):
        return self.tanh(x) * self.range + self.center


class ActorNetwork(nn.Module):
    
    def __init__(self, layers, actionspace_low, actionspace_high, activations=None, batch_norm=True, final_w=0.003):
        super(ActorNetwork, self).__init__()
        if activations == None:
            self.activations = [nn.ReLU6()]*(len(layers)-1)
        else:
            self.activations = activations

        self.clampactivation = ClampTanh(actionspace_low, actionspace_high)
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        self.use_batch_norm = batch_norm
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim_out) for dim_out in layers[1:-1]])

        # initialize weights
        for i in range(len(self.layers)-2):
            w = 1/ np.sqrt(layers[i])
            self.layers[i].weight.data = create_tens((layers[i+1], layers[i]), w)

        self.layers[-1].weight.data = create_tens((layers[-1], layers[-2]), final_w)

    def forward(self, x):
        output = x
        for i in range(len(self.layers)-1):
            output = self.layers[i](output)
            output = self.activations[i](output)
            if self.use_batch_norm:
                output = self.batch_norms[i](output)
        output = self.clampactivation(self.layers[-1](output))
        return output




    
