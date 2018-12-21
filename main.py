import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
import numpy as np

from DDPG import DDPG

def ballBalancerObsTransform(obs):
    if type(obs) is list or type(obs) is tuple:
        return obs[0]
    else:
        return obs

identity = lambda x: x

env = gym.make("CartpoleSwingShort-v0")

ddpg = DDPG(env=env, episodes=50, min_samples_during_trial=100, trial_horizon=500, transform=identity, noise_init=1., warmup_samples=5000, buffer_size=10000, noise_decay=0.95)

ddpg.train()
ddpg.actor_target.eval()

episodes = 100
rew = []

for step in range(episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        #transformation to action
        obs = ddpg.transformObservation(obs)
        state = torch.tensor(obs, dtype=torch.float32)
        
        action = ddpg.actor_target(state).detach().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if step >= episodes-10:
            env.render()

    rew.append(total_reward)

plt.plot(range(episodes), rew)
plt.show()
print(sum(rew)/len(rew))
