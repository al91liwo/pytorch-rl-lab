import torch


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