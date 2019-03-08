import torch
from ActorNetwork import ActorNetwork
import quanser_robots
import gym
import sys

env = gym.make("CartpoleSwingShort-v0")

layers = [env.observation_space.shape[0], 400, 500, env.action_space.shape[0]]

actor = ActorNetwork(layers, torch.tensor(-5.),torch.tensor(5.))
actor.load_state_dict(torch.load(sys.argv[1]))

done = False
obs = env.reset()
while not done:
    action = actor(torch.tensor(obs, dtype=torch.float32))
    obs, reward, done, _ = env.step(action.detach().numpy())
    env.render()