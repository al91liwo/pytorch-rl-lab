"""
MPC requires the reward function as an input.
This file defines the reward functions of different environments
and maps them
"""
import torch
import numpy as np
from src.utility.util import angle_from_sincos
from src.utility.util import angle_normalize


def perfect_reward_model_pendulum(state, action):
    """
    Calculates perfect reward for given state and taken action or a batch of states and actions
    :param state: the current state as tensor
    :param action: the action that is executed as tensor
    :return: reward for state action pair
    """
    # Original code in numpy
    # th = _angle_from_sincos(state[0], state[1])
    # thdot = state[2]
    #
    # u = np.clip(action[0], -2, 2)
    # costs = _angle_normalize(th) ** 2 \
    #         + .1 * thdot ** 2 \
    #         + .001 *(u ** 2)
    if state.dim == 1:  # we ave a single state action pair
        theta = angle_from_sincos(state[0], state[1])
        theta_dot = state[2]
        u = np.clip(action.item(), -2, 2)
        # cost = _angle_normalize(theta) ** 2 + .1 * theta_dot ** 2 + .001 * u ** 2
    else:  # we have a batch
        theta = angle_from_sincos(state[:, 0], state[:, 1])
        theta_dot = state[:, 2]
        u = torch.clamp(action.squeeze(), -2, 2)
    cost = angle_normalize(theta) ** 2 + .1 * theta_dot ** 2 + .001 * u ** 2
    return -cost


def perfect_reward_model_cartpole_stab(state, action):
    """
    Perfect reward model of the cartpole stabilization task in the quanser_robots environment
    :param state: the state to calculate the reward from
    :param action: needs to fit the interface is somebody wants to make an action based reward function
    :return: the reward
    """
    if state.dim == 1:
        cos_th = state[2]
    else:
        cos_th = state[:, 2]

    return -cos_th + 1.


def perfect_reward_model_cartpole_swing(state, action):
    """
    Perfect reward model of the cartpole swing up task in the quanser_robots environment
    :param state: the state to calculate the reward from
    :param action: needs to fit the interface if somebody wants to make an action based reward function
    :return: the reward
    """
    if state.dim == 1:
        cos_th = state[2]
    else:
        cos_th = state[:, 2]

    return -cos_th + 1.


rewards = {
    "Pendulum-v0": perfect_reward_model_pendulum,
    "CartpoleStabShort-v0": perfect_reward_model_cartpole_stab,
    "CartpoleSwingShort-v0": perfect_reward_model_cartpole_stab
}

# reward functions using only torch and not numpy
rewards_t = {
    "Pendulum-v0": perfect_reward_model_pendulum,
    "CartpoleStabShort-v0": perfect_reward_model_cartpole_stab,
    "CartpoleSwingShort-v0": perfect_reward_model_cartpole_stab
}