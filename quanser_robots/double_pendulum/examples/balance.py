import numpy as np
import gym
from quanser_robots.double_pendulum.ctrl import BalanceCtrl
import matplotlib.pyplot as plt

def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta

class PlotSignal:
    def __init__(self, window = 500):
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

            plt.subplot(N, 1, i+1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i+1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def main():
    plt.ion()
    env = gym.make('DoublePendulum-v0')                     # Use "DoublePendulumRR-v0" for the simulation
    window = 500
    real_plot = PlotSignal(window=window)
    obs = env.reset()
    ctrl = BalanceCtrl(dt=env.env.timing.dt)
    print("Reset done")

    use_plot = False                                 # Disable for the real system: it slows down
    collect_fr = 1                                    # Frequency collecting data
    plot_fr = 1                                       # Frequency refresh plot
    render = True                                      # Render true for visualizing the simulation
    render_fr = 10                                     # Render frequency: only for simulation

    i= 0
    while not ctrl.done and i < 60. / env.env.timing.dt:

        i += 1
        act = ctrl(obs)
        obs, _, _, _ = env.step(np.array(act[0]))

        if render:
            if i % render_fr == 0:
                env.render()

        if use_plot:

            if i % collect_fr == 0:
                x, theta1, theta2, x_dot, theta1_dot, theta2_dot = obs
                real_plot.update(theta1=theta1, theta1_dot=theta1_dot, theta2=theta2, theta2_dot=theta2_dot,volt=act[0],u=act[1], x=x)

            if i % plot_fr == 0:
                real_plot.plot_signal()

    print("Time %f s" % (i*env.env.timing.dt))
    env.step(np.array([0.]))
    env.close()
    print("Finish")


if __name__ == "__main__":
    main()
