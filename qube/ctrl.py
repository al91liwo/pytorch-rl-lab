import numpy as np
from .base import QubeDynamics


class PDCtrl:
    """
    Slightly tweaked PD controller (increases gains if `x_des` not reachable).

    Accepts `th_des` and drives Qube to `x_des = (th_des, 0.0, 0.0, 0.0)`

    Flag `done` is set when `|x_des - x| < tol`.

    Tweak: increase P-gain on `th` if velocity is zero but the goal is still
    not reached (useful for counteracting resistance from the power cord).
    """

    def __init__(self, K=None, th_des=0.0, tol=5e-2):
        self.done = False
        self.K = K if K is not None else [5.0, 0.0, 0.5, 0.0]
        self.th_des = th_des
        self.tol = tol

    def __call__(self, x):
        th, al, thd, ald = x
        K, th_des, tol = self.K, self.th_des, self.tol
        all_but_th_squared = al ** 2 + thd ** 2 + ald ** 2
        err = np.sqrt((th_des - th) ** 2 + all_but_th_squared)
        if not self.done and err < tol:
            self.done = True
        elif th_des and np.sqrt(all_but_th_squared) < tol / 5.0:
            # Increase P-gain on `th` when struggling to reach `th_des`
            K[0] += 0.01 * K[0]
        return K[0]*(th_des - th) - K[1] * al - K[2] * thd - K[3] * ald


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, x_init, positive=True):
        self.done = False
        self.th_lim = 0.0
        self.thd_max = 1e-4
        self.sign = 1 if positive else -1
        self.u_max = 1.2
        self.cnt = 0
        self.max_cnt = 500
        self.th_init = x_init[0]
        self.delta_th_min = 0.1

    def __call__(self, x):
        if self.cnt < self.max_cnt:
            self.cnt += 1
        else:
            th, _, thd, _ = x
            if self.sign * self.th_lim < self.sign * th:
                self.th_lim = th
            if np.abs(thd) < self.thd_max and \
                    np.abs(th - self.th_init) > self.delta_th_min:
                self.done = True
        return self.sign * self.u_max


class CalibrCtrl:
    """Go to joint limits, find midpoint, go to the midpoint."""

    def __init__(self, x_init):
        self.done = False
        self.go_right = GoToLimCtrl(x_init, positive=True)
        self.go_left = GoToLimCtrl(x_init, positive=False)
        self.go_center = PDCtrl()

    def __call__(self, x):
        u = 0.0
        if not self.go_right.done:
            u = self.go_right(x)
        elif not self.go_left.done:
            u = self.go_left(x)
        elif not self.go_center.done:
            if self.go_center.th_des == 0.0:
                self.go_center.th_des = \
                    (self.go_left.th_lim + self.go_right.th_lim) / 2
            u = self.go_center(x)
        elif not self.done:
            self.done = True
        return u


class EnergyCtrl:
    """PD controller on energy."""

    def __init__(self, mu, Er):
        self.mu = mu  # P-gain on the energy (m/s/J)
        self.Er = Er  # reference energy (J)
        dyn = QubeDynamics()
        self.Ek = lambda alpha_dot: 0.5 * dyn.Jp * alpha_dot ** 2
        pot_en_const = 0.5 * dyn.Mp * dyn.Lp * dyn.g
        self.Ep = lambda alpha: pot_en_const * (1.0 - np.cos(alpha))
        self.volt = lambda acc: -dyn.Mr * dyn.Lr * acc * dyn.Rm / dyn.kt

    def __call__(self, x):
        _, alpha, _, alpha_dot = x
        E = self.Ek(alpha_dot) + self.Ep(alpha)
        acc = self.mu * (self.Er - E) * np.sign(alpha_dot * np.cos(alpha))
        return self.volt(acc)


class SwingUpCtrl:
    """Hybrid controller (EnergyCtrl, PDCtrl) switching based on alpha."""

    def __init__(self, ref_energy=0.028, energy_gain=50.0,
                 alpha_max_pd_enable=20.0, pd_gain=None):
        # Set up the energy pumping controller
        self.en_ctrl = EnergyCtrl(mu=energy_gain, Er=ref_energy)
        # Set up the PD controller
        cos_al_delta = 1.0 + np.cos(np.pi - np.deg2rad(alpha_max_pd_enable))
        self.pd_enabled = lambda cos_al: np.abs(1.0 + cos_al) < cos_al_delta
        pd_gain = pd_gain if pd_gain is not None else [-1.5, 25.0, -1.5, 2.5]
        self.pd_ctrl = PDCtrl(K=pd_gain)

    def __call__(self, obs):
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
        x = np.r_[np.arctan2(sin_th, cos_th),
                  np.arctan2(sin_al, cos_al),
                  th_d, al_d]
        if self.pd_enabled(cos_al):
            x[1] = x[1] % (2 * np.pi) - np.pi
            return self.pd_ctrl(x)
        else:
            return self.en_ctrl(x)
