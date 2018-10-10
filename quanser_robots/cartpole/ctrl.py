import numpy as np
import time


class PDCtrl:
    """
    PD controller for the cartpole environment.
    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, s_des=np.zeros(4), tol=5e-4):
        self.K = K if K is not None else np.array([20.0, 0.0, 0.0, 0.0])

        self.done = False
        self.s_des = s_des
        self.tol = tol

    def __call__(self, s):

        # Compute the voltage:
        err = self.s_des - s
        v = np.dot(self.K.transpose(), err)

        # Check for completion:
        if np.sum(err**2) <= self.tol:
            self.done = True

        return np.array([v], dtype=np.float32)

class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, s_init, positive=True):
        self.done = False
        self.success = False
        self.x_init = s_init[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1

        self.sign = 1 if positive else -1
        self.u_max = self.sign * np.array([1.5])

        self._t_init = False
        self._t0 = 0.0
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, s):
        x, _, xd, _ = s

        # Initialize the time:
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

        # Compute voltage:
        if (time.time() - self._t0) < self._t_min:
            u = self.u_max

        elif np.abs(xd) < self.xd_max: # and np.abs(x - self.x_init) > self.delta_x_min:
            u = np.zeros(1)
            self.success = True
            self.done = True

        elif (time.time() - self._t0) > self._t_max:
            u = np.zeros(1)
            self.success = False
            self.done = True

        else:
            u = self.u_max

        return u