"""
Minimal communication interface -- just the step function.
"""

from common import QSocket, VelocityFilter
import numpy as np


class Qube:
    def __init__(self, ip="192.172.162.1"):
        self._filt = VelocityFilter(2)
        self._qsoc = QSocket(ip, 2, 1)

    def __call__(self, a):
        t, x = self._qsoc.snd_rcv(np.clip(a, -5.0, 5.0))
        s = np.r_[x, self._filt(x)]
        return t, s


if __name__ == "__main__":
    f = Qube()
    u = np.array([0.0])
    t, s = f(u)
    print(t, s)

