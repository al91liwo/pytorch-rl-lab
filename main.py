import gym
import numpy as np
import torch

import quanser_robots
from DDPG import DDPG

env = gym.make("Pendulum-v0")

ddpg = DDPG(env)

ddpg.train()

done = False
while True:
    obs = env.reset()
    while not done:
        #transformation to action
        obs = ddpg.transformObservation(obs)
        state = np.reshape(np.array(obs), (1, ddpg.state_dim))
        state = torch.from_numpy(state).type(torch.FloatTensor)
        action = ddpg.actor_network(state).item()
        # no scalar allowed => [action]
        obs, reward, done, _ = env.step([action])
        env.render()
        print(reward)
