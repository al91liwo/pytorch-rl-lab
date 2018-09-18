import gym
from qube.base import SwingUpCtrl

env = gym.make('Qube-v0')

ctrl = SwingUpCtrl()
obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, done, _ = env.step(act)
env.step(0.0)

env.close()
