import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG

env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)


ddpg = DDPG(env=env, episodes=50,
            actor_hidden_layers=[300, 400, 300], critic_hidden_layers=[300, 400, 300])

ddpg.train()
ddpg.actor_target.eval()

episodes = 100
rew = []

for step in range(episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        obs = ddpg.transformObservation(obs)
        state = torch.tensor(obs, dtype=torch.float32)
        
        action = ddpg.actor_target(state).detach().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if step == episodes-1:
            env.render()

    rew.append(total_reward)
env.close()

plt.plot(range(episodes), rew)
plt.show()
print(sum(rew)/len(rew))
ddpg.save_model(env_name)
