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

env = quanser_robots.GentlyTerminating(gym.make("Qube-v0"))

ddpg = DDPG(env=env, min_batches=1, min_samples_during_trial=20, transform=identity, noise_init=1., tau=1e-4, noise_decay=.95, warmup_noise=1., actor_lr=1e-4, warmup_samples=2000, buffer_size=10000, actor_hidden_layers=[100, 200, 300, 300], critic_hidden_layers=[100, 200, 300, 300])

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
