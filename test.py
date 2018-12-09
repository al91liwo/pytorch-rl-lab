import gym
import quanser_robots
import numpy as np
import torch
from torch import nn


class net(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.batch0 = nn.BatchNorm1d(self.input_dim)
        self.func0 = nn.Linear(self.input_dim, self.output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.t()
        print(x)
        x = self.batch0(x)
        print(x)
        x = self.func0(x)
        print(x)
        x = self.relu(x)
        return x


env = gym.make("Pendulum-v0")

obs = env.reset()

net = net(env.observation_space.shape[0], env.action_space.shape[0])

t = []
for i in range(10):
    arr = np.reshape(np.array(obs), (1,3))

    x = torch.from_numpy(arr).type(torch.FloatTensor)
    t.append(x)

t = torch.cat(t, dim=0)
print(t)
net.eval()
y = net(t)
print(y)





