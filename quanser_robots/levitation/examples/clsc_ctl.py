"""
An example of a custom controller implementation.
"""

import time
import numpy as np
import gym
import quanser_robots
from quanser_robots.levitation.ctrl import CurrentPICtrl

import scipy
from scipy import signal

import matplotlib.pyplot as plt


def currentCtl(ref):
    env = gym.make('Coil-v0')

    state = np.zeros((len(ref), 2))

    ctrl = CurrentPICtrl()
    obs = env.reset()

    for i in range(len(ref)):
        state[i, :] = obs
        act = ctrl(obs, sref=ref[i])
        obs, _, _, _ = env.step(act)

    env.close()

    return state


def gapCtl():
    env = gym.make('Levitation-v0')

    obs = env.reset()
    obsn, _, _, _ = env.step(np.array([1.0]))
    env.close()


if __name__ == "__main__":

    # Coil current control
    t = np.linspace(0, 1, 10000, endpoint=False)
    ref = np.clip(2.0 * signal.square(2 * np.pi * 5 * t), 0.0, 3.0)

    state = currentCtl(ref)

    plt.plot(state[:, 0])
    plt.plot(ref)
    plt.title('Coil Current Control')
    plt.xlabel('Step')
    plt.ylabel('Current')
    plt.legend(['Current', 'Reference'])

    # # Levitaiton simulation
    # gapCtl(verbose=True)
