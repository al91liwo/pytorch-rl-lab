"""
An example of a custom controller implementation.
"""

import time
import numpy as np
import gym
import quanser_robots
from quanser_robots.levitation.ctrl import CurrentPICtrl, GapPIVCtrl

import scipy
from scipy import signal

import matplotlib.pyplot as plt


def ratelimit(x,t,rlim):
    def helper():
        y = x[0]
        tprev = t[0]
        for (xi,ti) in zip(x,t):
            dy = xi - y
            dt = ti - tprev
            y += np.clip(dy,-rlim*dt,rlim*dt)
            tprev = ti
            yield y
    return np.array(list(helper()))


def currentCtl(ref):
    env = gym.make('Coil-v0')

    state = np.zeros((len(ref), 2))
    action = np.zeros((len(ref), 1))

    ctrl = CurrentPICtrl(wind=True)
    obs = env.reset()

    for i in range(len(ref)):
        act = ctrl(obs, sref=ref[i])

        state[i, :] = obs
        action[i, :] = act

        obs, _, _, _ = env.step(act)

    env.close()

    return state, action


def gapCtl(ref):
    env = gym.make('Levitation-v0')

    state = np.zeros((len(ref), 2))
    action = np.zeros((len(ref), 1))

    ctrl = GapPIVCtrl()
    obs = env.reset()

    for i in range(len(ref)):
        act = ctrl(obs, sref=ref[i])

        state[i, :] = obs
        action[i, :] = act

        obs, _, _, _ = env.step(act)

    env.close()

    return state, action


if __name__ == "__main__":

    # coil current control
    T, f = 10, 500
    t = np.linspace(0, T, f * T, endpoint=False)

    ref = np.clip(signal.square(2 * np.pi * 0.2 * t - np.pi), 0.0, 1.0)
    plt.figure()
    plt.subplot(211)
    plt.plot(t, ref)
    plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
    plt.xlabel('Time')
    plt.title('Coil Current Control')

    state, action = currentCtl(ref)
    plt.plot(t, state[:, 0])
    plt.ylabel('Current')
    plt.legend(['Reference', 'Current'])

    plt.subplot(212)
    plt.plot(t, action)
    plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
    plt.ylabel('Voltage')
    plt.xlabel('Time')
    plt.title('Coil Current Control')

    # levitaiton simulation
    T, f = 10, 500
    t = np.linspace(0, T, f * T, endpoint=False)

    # reproduce ref signal of Quanser
    ref = 0.014 + 1e-3 * signal.square(2 * np.pi * 0.25 * t - np.pi)
    ref[500 * 2: ] += 1.0 * (-0.006 + 1e-3)
    ref = ratelimit(ref, t, 0.005)
    plt.figure()
    plt.subplot(211)
    plt.plot(t, ref)
    plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
    plt.xlabel('Time')
    plt.title('Gap Control')

    state, action = gapCtl(ref)
    plt.plot(t, state[:, 0])
    plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
    plt.ylabel('Gap')
    plt.legend(['Reference', 'Gap'])

    plt.subplot(212)
    plt.plot(t, action)
    plt.xticks(np.arange(min(t), max(t) + 1, 1.0))
    plt.ylabel('Current')
    plt.xlabel('Time')
    plt.title('Gap Control')
