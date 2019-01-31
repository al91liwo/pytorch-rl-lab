import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG

env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)

ddpg = DDPG(env=env, episodes=100, warmup_samples=10000, buffer_size=25000, batch_size=32,
            actor_lr=1e-3, critic_lr=1e-3,
            actor_hidden_layers=[200, 100, 100], critic_hidden_layers=[200, 100, 100])

ddpg.train()
ddpg.actor_target.eval()

episodes = 500
rew = []

for step in range(episodes):
    done = False
    obs = env.reset()[0]
    total_reward = 0
    print(step)
    while not done:
        obs = ddpg.transformObservation(obs)
        state = torch.tensor(obs, dtype=torch.float32)
        
        action = ddpg.actor_target(state).detach().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()


    rew.append(total_reward)
env.close()

plt.plot(range(episodes), rew)
plt.show()
print(sum(rew)/len(rew))
ddpg.save_model(env_name)
