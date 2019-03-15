import torch
import math


def angle_normalize(x):
    """
    Normalizes angle on interval -pi to pi
    :param x: angle in radians
    :return: normalized angle in radians
    """
    return ((x+math.pi) % (2*math.pi)) - math.pi


def angle_from_sincos(cos, sin):
    """
    Calculates the angle from sin and cos
    :param sin: the sin of the angle
    :param cos: the cos of the angle
    :return: the angle in radiants
    """
    cos_angle = torch.acos(cos)
    sin_sign = sin
    sin_sign[sin > 0] = 1.
    sin_sign[sin < 0] = -1.
    return torch.abs(cos_angle) * sin_sign


def create_tens(input_dim, weight):
    """
    Utility function that creates tensors for
    the actor and critic networks
    param input_dim: input dimension of a layer
    param weight: weight under uniform distribution
    return: tensor of size input_dim with weights under uniform distribution
    """
    return torch.zeros(input_dim, dtype=torch.float).uniform_(-weight, weight)


def validate_config(config, layout):
    """
    Checks if the given config satifies the developers layout
    :param config:
    :param layout:
    :return:
    """
    layout = layout

    for c in config:
        if not c in layout:
            raise Exception(c + " is no valid parameter, please check your layout or configuration")
        layout[c] = config[c]