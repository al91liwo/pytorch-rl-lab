import gym
from quanser_robots.qube import SwingUpCtrl

env = gym.make('Qube-v0')

ctrl = SwingUpCtrl()
obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, done, _ = env.step(act)

env.close()
