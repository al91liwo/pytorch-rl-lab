import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG

env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)

ddpg = DDPG(env=env, episodes=2000, warmup_samples=50000, buffer_size=500000, batch_size=32,
            actor_lr=1e-3, critic_lr=1e-3, noise_decay=0.99,
            actor_hidden_layers=[200, 200, 100, 100], critic_hidden_layers=[200, 200, 100, 100])

ddpg.train()
ddpg.actor_target.eval()

episodes = 100
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

        if step >= episodes - 1:
            env.render()


    rew.append(total_reward)
env.close()


print(sum(rew)/len(rew))
ddpg.save_model(env_name)
plt.plot(range(episodes), rew)
plt.savefig('models/{}/over100episodes.jpg'.format(ddpg.started))
plt.show()