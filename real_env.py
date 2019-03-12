import gym
import torch
from quanser_robots import GentlyTerminating
import sys
from src.algorithm.DDPG.ActorNetwork import ActorNetwork
import numpy as np
import pandas

torch.manual_seed(0)
env_name = "CartpoleStabRR-v0"
env = GentlyTerminating(gym.make(env_name))
# env = gym.make(env_name)

file = sys.argv[1]

ddpg_actor_params = torch.load(file)
ddpg_actor = ActorNetwork([env.observation_space.shape[0], 100, 100, 50, env.action_space.shape[0]], torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))
ddpg_actor.load_state_dict(ddpg_actor_params)
rew = []
total_reward = 0
obs = env.reset()
done = False
episodes = 1
trajectory = []
while not done:

    state = torch.tensor(obs, dtype=torch.float32)
    action = ddpg_actor(state).detach().numpy()
    action = np.clip(action, -5, 5)
    prev_obs = obs
    obs, reward, done, _ = env.step(action)
    trajectory.append([prev_obs, action, obs, reward, done])
    total_reward += reward

    rew.append(total_reward)
    # env.render()

env.close()
print(sum(rew) / len(rew))
print(total_reward)
df = pandas.DataFrame(trajectory, columns=['s', 'a', 'r', 'd', 's_n'])
df.to_csv("rrtrajectory/"+env_name+"_"+str(int(total_reward)))
# plt.xlabel("episodes")
# plt.ylabel("reward")
# plt.plot(range(episodes), rew)
# plt.show()
