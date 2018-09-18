import time
import numpy as np
import gym


class QubeBase(gym.Env):
    # Sampling and integration
    max_fs = 500.0  # max sampling freq
    fs = 50.0  # sampling frequency
    cmd_dur = int(max_fs / fs)  # repeat command `cmd_dur` times
    n_int_steps = max(int(max_fs / fs), 1)  # integration steps per command
    dt_int = 1.0 / (n_int_steps * fs)  # time step of the integrator

    # Motor
    Rm = 8.4  # resistance
    kt = 0.042  # current-torque (N-m/A)
    km = 0.042  # back-emf constant (V-s/rad)

    # Rotary arm
    Mr = 0.095  # mass (kg)
    Lr = 0.085  # length (m)
    Jr = Mr * Lr ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dr = 0.0  # equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum link
    Mp = 0.024  # mass (kg)
    Lp = 0.129  # length (m)
    Jp = Mp * Lp ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dp = 0.0  # equivalent viscous damping coefficient (N-m-s/rad)

    # Constants for equations of motion
    g = 9.81  # gravity
    a1 = Mp * Lr ** 2
    a2 = 0.5 * Mp * Lp ** 2
    a3 = 0.5 * Mp * Lp * Lr
    a4 = Jp + a2 / 2
    a5 = 0.5 * Mp * Lp * g

    # Limits
    state_max = np.array([2.3, np.inf, np.inf, np.inf])
    obs_max = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf])
    act_max = np.array([5.0])

    # Labels
    state_labels = ('theta', 'alpha', 'theta_dot', 'alpha_dot')
    obs_labels = ('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d')
    act_labels = ('volts',)
    
    def __init__(self):
        super(QubeBase, self).__init__()
        self.state_space = gym.spaces.Box(low=-self.state_max,
                                          high=self.state_max,
                                          dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.obs_max,
                                                high=self.obs_max,
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-self.act_max,
                                           high=self.act_max,
                                           dtype=np.float32)
        self._state = None

    def _snd_rcv(self, a):
        raise NotImplementedError
    
    def step(self, a):
        cost = 10.0 * (self._state[1] % (2 * np.pi) - np.pi) ** 2 \
               + 0.1 * self._state[3] ** 2 + 0.001 * a ** 2
        self._state, act = self._snd_rcv(a)
        obs = np.r_[np.cos(self._state[0]), np.sin(self._state[0]),
                    np.cos(self._state[1]), np.sin(self._state[1]),
                    self._state[2], self._state[3]]
        return obs, -cost / self.fs, False, {'s': self._state, 'a': act}


class PDCtrl:
    """
    Slightly tweaked PD controller (increases gains if `x_des` not reachable).

    Accepts `th_des` and drives QUBE to `x_des = (th_des, 0.0, 0.0, 0.0)`

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
        K, th_des, tol = self.K, self.th_des, self.tol
        all_but_th_squared = x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        err = np.sqrt((th_des - x[0]) ** 2 + all_but_th_squared)
        if not self.done and err < tol:
            self.done = True
        elif th_des and np.sqrt(all_but_th_squared) < tol / 5.0:
            # Increase P-gain on `th` when struggling to reach `th_des`
            K[0] += 0.01 * K[0]
        return K[0]*(th_des - x[0]) - K[1] * x[1] - K[2] * x[2] - K[3] * x[3]


class MetronomeCtrl:
    """
    Rhythmically swinging metronome.

    Example
    -------
        qube.run(MetronomeCtrl())

    """

    def __init__(self, u_max=5.0, f=0.5, dur=5.0):
        """
        Constructor

        :param u_max: maximum voltage
        :param f: frequency in Hz
        :param dur: task finishes in `dur` seconds

        """
        self.done = False
        self.u_max = u_max
        self.f = f
        self.dur = dur
        self.start_time = None

    def __call__(self, x):
        """
        Calculates the actions depending on the elapsed time.

        :return: scaled sinusoidal voltage
        """
        if self.start_time is None:
            self.start_time = time.time()
        t = time.time() - self.start_time
        if not self.done and t > self.dur:
            self.done = True
            u = 0.0
        else:
            u = 0.1 * self.u_max * np.sin(2 * np.pi * self.f * t)
        return u


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, positive=True):
        self.done = False
        self.th_lim = 0.0
        self.thd_max = 1e-4
        self.sign = 1 if positive else -1
        self.u_max = 0.8
        self.cnt = 0
        self.max_cnt = 10

    def __call__(self, x):
        if self.cnt < self.max_cnt:
            self.cnt += 1
        else:
            if self.sign * self.th_lim < self.sign * x[0]:
                self.th_lim = x[0]
            if np.abs(x[2]) < self.thd_max:
                self.done = True
        return self.sign * self.u_max


class CalibrCtrl:
    """Go to joint limits, find zero, and drive QUBE to the zero position."""

    def __init__(self):
        self.done = False
        self.go_right = GoToLimCtrl(positive=True)
        self.go_left = GoToLimCtrl(positive=False)
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
    """Nonlinear energy shaping controller for a swing up."""

    def __init__(self, mu=50.0, Er=0.025):
        self.mu = mu  # P-gain on the energy (m/s/J)
        self.Er = Er  # reference energy (J)

    def __call__(self, x):
        alpha, alpha_dot = x[1], x[3]
        Ek = 0.5 * QubeBase.Jp * alpha_dot ** 2
        Ep = QubeBase.a5 * (1 - np.cos(alpha))
        E = Ek + Ep
        a = self.mu * (self.Er - E) * np.sign(alpha_dot * np.cos(alpha))
        trq = QubeBase.Mr * QubeBase.Lr * a
        voltage = -QubeBase.Rm / QubeBase.kt * trq
        return voltage


class SwingUpCtrl:
    """Use EnergyCtrl for swing-up and switch to PDCtrl for stabilization."""

    def __init__(self, alpha_max_deg=20.0,
                 pd_ctrl=PDCtrl(K=[-1.5, 25.0, -1.5, 2.5]),
                 en_ctrl=EnergyCtrl()):
        self.cos_al_delta = 1.0 + np.cos(np.pi - np.deg2rad(alpha_max_deg))
        self.pd_ctrl = pd_ctrl
        self.en_ctrl = en_ctrl

    def __call__(self, obs):
        cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
        x = np.r_[np.arctan2(sin_th, cos_th),
                  np.arctan2(sin_al, cos_al),
                  th_d, al_d]

        if np.abs(cos_al + 1.0) < self.cos_al_delta:
            x[1] = x[1] % (2 * np.pi) - np.pi
            return self.pd_ctrl(x)
        else:
            return self.en_ctrl(x)
