import numpy as np
import matplotlib.pyplot as plt
import gym
from qube.base import SwingUpCtrl

plt.style.use('seaborn')

env = gym.make('Qube-v0')

ctrl = SwingUpCtrl()
obs = env.reset()
s_all, a_all = [], []
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, rwd, done, info = env.step(act)
    s_all.append(info['s'])
    a_all.append(info['a'])
env.step(0.0)

env.close()


fig, axes = plt.subplots(5, 1, figsize=(5, 8), tight_layout=True)

s_all = np.stack(s_all)
a_all = np.stack(a_all)

n_points = s_all.shape[0]
t = np.linspace(0, n_points / env.unwrapped._fs_ctrl, n_points)
for i in range(4):
    axes[i].plot(t, s_all.T[i], label=env.unwrapped.state_labels[i], c=f'C{i}')
    axes[i].legend(loc='lower right')
axes[4].plot(t, a_all.T[0], label=env.unwrapped.act_labels[0], c=f'C{4}')
axes[4].legend(loc='lower right')

axes[0].set_ylabel('ang pos [rad]')
axes[1].set_ylabel('ang pos [rad]')
axes[2].set_ylabel('ang vel [rad/s]')
axes[3].set_ylabel('ang vel [rad/s]')
axes[4].set_ylabel('voltage [V]')
axes[4].set_xlabel('time [seconds]')
plt.show()
