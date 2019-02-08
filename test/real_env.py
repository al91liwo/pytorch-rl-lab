import gym
import torch
import matplotlib.pyplot as plt
from quanser_robots import GentlyTerminating
import os
import sys

torch.manual_seed(0)
env_name = "CartpoleStabRR-v0"
env = GentlyTerminating(gym.make(env_name))


file = sys.argv[1]

ddpg_actor = torch.load(os.path(file))

rew = []
total_reward = 0
obs = env.reset()
done = False
episodes = 1
while not done:

    state = torch.tensor(obs, dtype=torch.float32)

    action = ddpg_actor(state).detach().numpy()
    obs, reward, done, _ = env.step(action)
    total_reward += reward

    rew.append(total_reward)

env.close()

print(sum(rew) / len(rew))
plt.xlabel("episodes")
plt.ylabel("reward")
plt.plot(range(episodes), rew)
plt.show()
