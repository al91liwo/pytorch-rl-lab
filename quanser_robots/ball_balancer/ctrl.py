import time
import numpy as np


class DummyCtrl:
    """
    Randomly sampling actions for a specific time period.
    """
    def __init__(self, action_space, duration, action_scaling=0.1):
        self.done = False
        self.action_space = action_space
        self.action_scaling = action_scaling
        self.start_time = time.time()
        self.duration = duration  # time in s

    def __call__(self, _):
        if time.time() - self.start_time <= self.duration:
            action = self.action_space.sample() * self.action_scaling
        else:
            action = np.zeros(self.action_space.shape)
        return action


class PDCtrl:
    """
    Slightly tweaked PD controller (increases gains if `x_des` not reachable).

    Accepts `th_des` and drives Qube to `x_des = (th_des, 0.0, 0.0, 0.0)`

    Flag `done` is set when `|x_des - x| < tol`.

    Tweak: increase P-gain on `th` if velocity is zero but the goal is still
    not reached (useful for counteracting resistance from the power cord).
    """

    def __init__(self, kp=None, kd=None):
        """
        :param kp: constant controller feedback coefficients for error [V/m]
        :param kd: constant controller feedback coefficients for error time derivative [Vs/m]
        """
        self.Kp = np.zeros([14., 14.]) if kp is None else np.diag(kp)
        self.Kd = np.zeros([0., 0.]) if kd is None else np.diag(kd)

    def __call__(self, s, x_des=0., y_des=0.):
        """
        Calculate the controller output. u = -K*x
        :param s: state measurement
        :param x_des: goal position [m]
        :param y_des: goal position [m]
        :return: action [V]
        """
        th_x, th_y, x, y, th_x_dot, th_y_dot, x_dot, y_dot = s

        err = np.array([x_des - x, y_des - y])
        err_dot = np.array([0. - x_dot, 0. - y_dot])

        u = -1. * (self.Kp.dot(err) + self.Kd.dot(err_dot))
        return -u  # ask Quanser, why * -1
