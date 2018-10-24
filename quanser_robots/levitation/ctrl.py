import numpy as np


class CurrentPICtrl:
    """
    PI coil current controller with Anti-Windup

    Accepts `ic_des` and drives current to `x_des = (ic_des, 0.0)`
    """

    def __init__(self, K=None, dt=0.002, bsp=1.0,
                 act_sat=24.0, int_sat=24.0,
                 wind=False):

        self.K = K if K is not None else [230, 50430]
        self.dt = dt

        self.bsp = bsp

        self.act_sat = act_sat
        self.int_sat = int_sat

        self.mem = 0.0

        self.wind = wind
        self.awp = 0.0

    def __call__(self, s, sref):
        ic, icd = s
        ic_ref = sref
        K = self.K

        prop_err = (self.bsp * ic_ref - ic)
        prop_ctl = K[0] * prop_err

        int_err = (ic_ref - ic) + self.wind * 1.0 / K[0] * self.awp
        self.mem += int_err * self.dt
        self.mem = np.clip(self.mem, -self.int_sat, self.int_sat)
        int_ctl = K[1] * self.mem

        act = prop_ctl + int_ctl
        act_sat = np.clip(act, -self.act_sat, self.act_sat)

        self.awp = act_sat - act

        return np.array([act_sat])


class GapPIVCtrl:
    """
    PIV gap controller

    Accepts `xb_des` and drives Ball to `x_des = (xb_des, 0.0)`
    """

    def __init__(self, K=None, dt=0.002, bsp=1.0,
                 act_sat=3.0, int_sat=1.0):

        self.K = K if K is not None else [-194.0, -489.5, 143.0, -2.47]
        self.dt = dt

        self.bsp = bsp

        self.act_sat = act_sat
        self.int_sat = int_sat

        self.mem = 0.0

    def __call__(self, s, sref):
        xb, xbd = s
        xb_ref = sref
        K = self.K

        prop_err = (self.bsp * xb_ref - xb)
        prop_ctl = K[0] * prop_err

        int_err = (xb_ref - xb)
        self.mem += int_err * self.dt
        self.mem = np.clip(self.mem, -self.int_sat, self.int_sat)
        int_ctl = K[1] * self.mem

        ff_ctl =  xb_ref * self.K[2]
        vel_ctl = xbd * self.K[3]

        act = prop_ctl + int_ctl + 0.8 * ff_ctl - vel_ctl
        act_sat = np.clip(act, 0.0, self.act_sat)

        return np.array([act_sat])
