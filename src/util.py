import numpy
import torch

def create_tens(input_dim, weight):
    """
    Utility function that creates tensors for
    the actor and critic networks
    param input_dim: input dimension of a layer
    param weight: weight under uniform distribution
    return: tensor of size input_dim with weights under uniform distribution
    """
    return torch.zeros(input_dim).uniform_(-weight, weight)
