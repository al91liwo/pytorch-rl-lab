import numpy as np
import gym
from quanser_robots.cartpole.ctrl import SwingupCtrl, SwingdownCtrl
import matplotlib.pyplot as plt


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class PlotSignal:
    def __init__(self, window=10000):
        self.window = window
        self.values = {}

    def update(self, **argv):
        for k in argv:
            if k not in self.values:
                self.values[k] = [argv[k]]
            else:
                self.values[k].append(argv[k])
            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def do_trajectory(env, ctrl, plot, time_steps=2000, use_plot=True,
                  collect_fr=10, plot_fr=10, render=True, render_fr=True):

    obs = env.reset()
    for n in range(time_steps):
        act = ctrl(obs)
        obs, _, done, _ = env.step(np.array([act[0]]))

        # if done:
        #     print("Reach end")
        #     break

        if render:
            if n % render_fr == 0:
                env.render()

        if use_plot:

            if n % collect_fr == 0:
                alpha, theta = get_angles(obs[1], obs[2])
                plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=act[0], u=act[1], x=obs[0])
                env.render()

            if n % plot_fr == 0:
                plot.plot_signal()


def main():
    plt.ion()
    # Change from 'CartpoleSwing' to 'CartpoleStab' for task of swinging up or for only stabilization
    env = gym.make('CartpoleSwing-v0')  # Use "Cartpole-v0" for the simulation
    window = 500
    plot = PlotSignal(window=window)
    ctrl = SwingupCtrl(long=False)  # Use long=True if you are using the long pole

    do_trajectory(env, ctrl, plot, use_plot=False)

    env.step(np.array([0.]))
    env.close()
    input("Finish")


if __name__ == "__main__":
    main()
