import numpy as np
import matplotlib.pyplot as plt
import gym

from quanser_robots import GentlyTerminating
from quanser_robots.ball_balancer.ctrl import QPDCtrl


if __name__ == "__main__":
    env = GentlyTerminating(gym.make('BallBalancerRR-v0'))
    ctrl = QPDCtrl()

    obs, done = env.reset()
    obs_hist = [obs]
    act_hist = []
    rew_hist = []

    while not done:
        env.render()
        act = ctrl(obs)
        obs, rew, done, _ = env.step(act)
        act_hist.append(act)
        obs_hist.append(obs)
        rew_hist.append(rew)
    env.close()

    # Visualization
    fig, axes = plt.subplots(6, 1, figsize=(6, 8), tight_layout=True)

    obs_hist = np.stack(obs_hist)
    act_hist = np.stack(act_hist)
    rew_hist = np.stack(rew_hist)

    n_points = obs_hist.shape[0]
    t = np.linspace(0, n_points * env.unwrapped.timing.dt, n_points)  # TODO dt or dt_ctrl?
    for i in range(4):
        state_labels = env.unwrapped.state_space.labels[i]
        axes[i].plot(t, obs_hist.T[i], label=state_labels, c='C{}'.format(i))
        axes[i].legend(loc='lower right')

    action_labels = env.unwrapped.action_space.labels
    axes[4].plot(t[1:], act_hist[:, 0], label=action_labels[0], c='C{}'.format(4))
    axes[4].legend(loc='lower right')
    axes[5].plot(t[1:], act_hist[:, 1], label=action_labels[1], c='C{}'.format(4))
    axes[5].legend(loc='lower right')

    axes[0].set_ylabel('th_x [rad]')
    axes[1].set_ylabel('th_y [rad]')
    axes[2].set_ylabel('x [m]')
    axes[3].set_ylabel('y [m]')
    axes[4].set_ylabel('V_x [V]')
    axes[5].set_ylabel('V_y[V]')
    axes[5].set_xlabel('time')
    plt.show()
