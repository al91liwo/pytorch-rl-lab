import numpy as np


class PICtrl:
    """
    PI controller

    Accepts `ic_des` and drives Qube to `x_des = (ic_des, 0.0)`

    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, bsp=0.0, dt=0.002, ic_des=0.0, tol=5e-2):
        self.done = False
        self.K = K if K is not None else [230, 50430]
        self.bsp = bsp
        self.dt = dt

        self.ic_des = ic_des
        self.tol = tol

        self.mem = 0.0

    def __call__(self, x):
        ic, dic = x
        K, ic_des, tol = self.K, self.ic_des, self.tol

        err = np.sqrt((ic_des - ic) ** 2 + dic**2)
        if not self.done and err < tol:
            self.done = True

        self.mem += (ic_des - ic) * self.dt

        p_ctl = K[0] * (self.bsp * ic_des - ic)
        i_ctl = K[1] * self.mem

        return [p_ctl + i_ctl]
