import gym
import numpy as np
import torch

import quanser_robots
from DDPG import DDPG

env = gym.make("Pendulum-v2")

ddpg = DDPG(env)

ddpg.train()
ddpg.actor_target.eval()
while True:
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        #transformation to action
        obs = ddpg.transformObservation(obs)
        state = np.reshape(np.array(obs), (1, ddpg.state_dim))
        state = torch.from_numpy(state).type(torch.FloatTensor)
        
        action = ddpg.actor_target(state).item()
        # no scalar allowed => [action]
        print(action)
        obs, reward, done, _ = env.step([action])
        env.render()
        total_reward += reward
    print(total_reward)
    

       