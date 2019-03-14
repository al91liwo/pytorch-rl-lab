import torch, math

def angle_normalize(x):
    """
    Normalizes angle on interval -pi to pi
    :param x: angle in radians
    :return: normalized angle in radians
    """
    return (((x+math.pi) % (2*math.pi)) - math.pi)

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