import gym
import torch
import matplotlib.pyplot as plt

from DDPG import DDPG

env = gym.make("Pendulum-v0")


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ddpg = DDPG(env=env, state_dim=state_dim, action_dim=action_dim, episodes=1000)

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
        
        action = ddpg.actor_target(state).item()
        obs, reward, done, _ = env.step([action])
        total_reward += reward
        if step == episodes-1:
            env.render()

    rew.append(total_reward)
env.close()

plt.plot(range(episodes), rew)
plt.show()
print(sum(rew)/len(rew))
