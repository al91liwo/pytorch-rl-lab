import gym
import quanser_robots
import pandas as pd
from datetime import datetime

environment = "QubeRR-v0"

env = quanser_robots.GentlyTerminating(gym.make(environment))

columnNames = ['theta', 'sin_theta', 'cos_theta', 'x_dot', 'theta_dot', 'action', 'reward', 'theta\'', 'sin_theta\'', 'cos_theta\'', 'x_dot\'', 'theta_dot\'']

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
