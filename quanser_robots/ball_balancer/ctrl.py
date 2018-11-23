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


class QPDCtrl:
    """
    Reconstructing Quanser's PD-controller (see. q_2dbb_quick_start.mdl)
    """
    def __init__(self, kp=None, kd=None):
        """
        :param kp: constant controller feedback coefficients for error [V/m]
        :param kd: constant controller feedback coefficients for error time derivative [Vs/m]
        """
        self.Kp = np.diag([3.45, 3.45]) if kp is None else np.diag(kp)
        self.Kd = np.diag([2.11, 2.11]) if kd is None else np.diag(kd)
        self.limit_rad = 0.52360  # limit for angle command; see the saturation bock in the referenced mdl-file
        self.kp_servo = 14.  # P-control for servo angle; see the saturation bock in the referenced mdl-file

    def __call__(self, obs, x_des=0., y_des=0.):
        """
        Calculate the controller output.
        :param obs: state measurement a.k.a. observation
        :param x_des: goal position [m]
        :param y_des: goal position [m]
        :return: action [V]
        """
        th_x, th_y, x, y, th_x_dot, th_y_dot, x_dot, y_dot = obs

        err = np.array([x_des - x, y_des - y])
        err_dot = np.array([0. - x_dot, 0. - y_dot])
        th_des = self.Kp.dot(err) + self.Kd.dot(err_dot)

        # Saturation for desired angular position
        th_des = np.clip(th_des, -self.limit_rad, self.limit_rad)
        err_th = th_des - np.array([th_x, th_y])
        u = err_th * self.kp_servo

        return u  # see "Actuator Electrical Dynamics" block in referenced mdl-file
