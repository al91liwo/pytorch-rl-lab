"""
The minimal program that shows the basic control loop on the simulated swing-up.
"""

import gym
from quanser_robots import GentlyTerminating
from quanser_robots.qube import SwingUpCtrl

env = GentlyTerminating(gym.make('Qube-v0'))

ctrl = SwingUpCtrl(ref_energy=0.035, energy_gain=70.0, acc_max=7.0)
obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, done, _ = env.step(act)

env.close()
