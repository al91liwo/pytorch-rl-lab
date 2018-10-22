import numpy as np


class CurrentPICtrl:
    """
    PI coil current controller with Anti-Windup

    Accepts `ic_des` and drives current to `x_des = (ic_des, 0.0)`
    """

    def __init__(self, K=None, bsp=1.0, dt=0.002, sat=24.0):
        self.K = K if K is not None else [230, 50430]
        self.bsp = bsp
        self.dt = dt
        self.sat = sat

        self.mem = 0.0
        self.awp = 0.0

    def __call__(self, s, sref):
        ic, icd = s
        ic_ref = sref
        K = self.K

        prop_err = (self.bsp * ic_ref - ic)
        prop_ctl = K[0] * prop_err

        int_err = (ic_ref - ic) + 1.0 / K[0] * self.awp
        self.mem += int_err * self.dt
        self.mem = np.clip(self.mem, -self.sat, self.sat)
        int_ctl = K[1] * self.mem

        act = prop_ctl + int_ctl
        act_sat = np.clip(act, -self.sat, self.sat)
        self.awp = act_sat - act

        return np.array([act_sat])


class GapPIVCtrl:
    """
    PIV gap controller with Anti-Windup

    Accepts `xb_des` and drives Ball to `x_des = (xb_des, 0.0)`
    """

    def __init__(self, K=None, bsp=1.0, dt=0.002, sat=3.0):
        self.K = K if K is not None else [-194.0, -489.5, 143.0, -2.47]
        self.bsp = bsp
        self.dt = dt
        self.sat = sat

        self.mem = 0.0

        self.done = False

    def __call__(self, s, sref):
        xb, xbd = s
        xb_ref = sref
        K = self.K

        prop_err = (self.bsp * xb_ref - xb)
        prop_ctl = K[0] * prop_err

        int_err = (xb_ref - xb)
        self.mem += int_err * self.dt
        self.mem = np.clip(self.mem, -1.0, 1.0)
        int_ctl = K[1] * self.mem

        ff_ctl =  xb_ref * self.K[2]
        vel_ctl = xbd * self.K[3]

        act = prop_ctl + int_ctl + ff_ctl - vel_ctl
        act_sat = np.clip(act, 0.0, self.sat)

        return np.array([act_sat])
