"""
An example of a custom controller implementation.
"""

import time
import numpy as np
import gym
import quanser_robots
from quanser_robots.levitation.ctrl import PICtrl

def main():
    env = gym.make('Levitation-v0')

    ctrl = PICtrl(ic_des=env.unwrapped.goal)
    obs = env.reset()
    while not ctrl.done:
        # env.render()
        act = ctrl(obs)
        obs, _, _, _ = env.step(act)
        # print(obs)
    env.close()


if __name__ == "__main__":
    main()
