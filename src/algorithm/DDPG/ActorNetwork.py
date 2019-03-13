import numpy as np
from torch import nn

from src.util import create_tens


class ClampTanh(nn.Module):
    def __init__(self, low, high):
        """
        ClampTanh is a customizable tanh activation function.
        It is a tanh function that is sized to output values from low to high.
        :param low: lowest value that will be clamped
        :param high: highest value that will be clamped
        """
        super(ClampTanh, self).__init__()
        self.tanh = nn.Tanh()
        if low >= high:
            print("low should be smaller than high!")
        self.range = (high - low)/2
        self.center = (low + high)/2

    def forward(self, x):
        """
        Given an input x that will be clamped between self.low and self.high
        :param x: given input or batch
        :return: clamped action in ActorNetwork
        """
        return self.tanh(x) * self.range + self.center


class ActorNetwork(nn.Module):
    
    def __init__(self, layers, actionspace_low, actionspace_high, activations=None, batch_norm=True, final_w=0.003):
        """
        Actor Network for ddpg algorithm as specified in the paper (link in DDPG class)
        :param layers: numeric list of layers that will be used in this network
        :param actionspace_low: lowest action that can be used
        :param actionspace_high: highest action that can  be used
        :param activations: list of activation functions that fullfill length of all layers
        :param batch_norm: either use or use not batch norm for this network
        :param final_w: weight initialization for the output layer
        """
        super(ActorNetwork, self).__init__()
        if activations == None:
            self.activations = [nn.ReLU6()]*(len(layers)-1)
        else:
            self.activations = activations

        self.clampactivation = ClampTanh(actionspace_low, actionspace_high)
        print(layers)
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        self.use_batch_norm = batch_norm
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim_out) for dim_out in layers[1:-1]])

        # initialize weights
        for i in range(len(self.layers)-2):
            w = 1/ np.sqrt(layers[i])
            self.layers[i].weight.data = create_tens((layers[i+1], layers[i]), w)

        self.layers[-1].weight.data = create_tens((layers[-1], layers[-2]), final_w)

    def forward(self, x):
        """
        Forwarding state sthrough actor_network
        :param x: states either as a batch or a single state
        :return: yields the action thtat will be used
        """
        output = x
        for i in range(len(self.layers)-1):
            output = self.layers[i](output)
            output = self.activations[i](output)
            if self.use_batch_norm:
                output = self.batch_norms[i](output)
        output = self.clampactivation(self.layers[-1](output))
        return output




    
