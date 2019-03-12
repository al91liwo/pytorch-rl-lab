import torch
from src.algorithm.DDPG.ActorNetwork import ActorNetwork
import gym
import sys
import numpy as np

env = gym.make("CartpoleSwingShort-v0")

layers = [env.observation_space.shape[0], 300, 300, 300, env.action_space.shape[0]]

actor = ActorNetwork(layers, torch.tensor(-10.),torch.tensor(10.))
actor.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
actor.eval()
done = False
obs = env.reset()
total_reward = 0
while not done:
    state = torch.tensor(obs, dtype=torch.float32)
    action = [actor(state.unsqueeze(0)).squeeze().cpu().detach().numpy()]
    obs, reward, done, _ = env.step(np.array(action))
    env.render()
    total_reward += reward
print(total_reward)
