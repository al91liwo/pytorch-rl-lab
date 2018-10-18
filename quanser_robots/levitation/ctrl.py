import numpy as np


class PICtrl:
    """
    PI controller

    Accepts `ic_des` and drives Qube to `x_des = (ic_des, 0.0)`

    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, ic_des=0.0, tol=5e-2):
        self.done = False
        self.K = K if K is not None else [1.0, 0.1]
        self.ic_des = ic_des
        self.tol = tol

        self.mem = 0.0

    def __call__(self, x):
        ic, dic = x
        K, ic_des, tol = self.K, self.ic_des, self.tol

        err = np.sqrt((ic_des - ic) ** 2 + dic**2)
        if not self.done and err < tol:
            self.done = True

        self.mem += (ic_des - ic)

        return [K[0]*(ic_des - ic) + K[1]*self.mem]
