import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
import pandas as pd
from datetime import datetime

from DDPG import DDPG

environment = "Pendulum-v2"

env = gym.make(environment)

columnNames = ['theta', 'theta_dot', 'action', 'reward', 'theta\'', 'theta_dot\'']
data = []
episodes = 100

state = env.reset()
for step in range(episodes):
    action = env.action_space.sample()
    nextState, reward, done, info = env.step(action)
    data.append([*state, *action, reward, *nextState])
    state = nextState

dataFrame = pd.DataFrame(data, columns=columnNames)
dataFrame.to_csv(environment +"_"+ datetime.now().strftime('%Y-%m-%d %H:%M:%S') +".csv", index=False)