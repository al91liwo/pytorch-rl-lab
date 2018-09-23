import numpy as np
import gym


class QubeBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(QubeBase, self).__init__()
        self._state = None
        self.timing = Timing(fs, fs_ctrl)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Limits
        safety_th_lim = 1.5
        act_max = np.array([5.0])
        state_max = np.array([2.0, 6.0 * np.pi, 30.0, 40.0])
        sens_max = np.array([2.3, np.inf])
        obs_max = np.array([np.cos(state_max[0]), np.sin(state_max[0]),
                            1.0, 1.0, state_max[2], state_max[3]])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('theta', 'alpha'),
            low=-sens_max, high=sens_max, dtype=np.float32)
        self.state_space = LabeledBox(
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('cos_th', 'sin_th', 'cos_al', 'sin_al', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)

        # Function to ensure that state and action constraints are satisfied
        self._lim_act = ActionLimiter(self.state_space,
                                      self.action_space,
                                      safety_th_lim)

    def _sim_step(self, x, a):
        raise NotImplementedError

    def _ctrl_step(self, a):
        x = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = self._lim_act(x, a)
            x = self._sim_step(x, a_cmd)
        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, x, a):
        th, al, thd, ald = x
        al_wrap = al % (2 * np.pi) - np.pi
        cost = al_wrap**2 + 5e-3*ald**2 + 1e-2*th**2 + 2e-2*thd**2 + 1e-3*a**2
        done = not self.state_space.contains(x)
        rwd = np.exp(-cost) * self.timing.dt_ctrl
        return rwd, done

    def step(self, a):
        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        obs = np.r_[np.cos(self._state[0]), np.sin(self._state[0]),
                    np.cos(self._state[1]), np.sin(self._state[1]),
                    self._state[2], self._state[3]]
        return obs, rwd, done, {'s': self._state, 'a': act}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):
        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = \
            2.0 * action_space.high[0] / (self._th_lim_max - self._th_lim_min)
        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)

    def _joint_lim_violation_force(self, x):
        th, _, thd, _ = x
        up = self._relu(th-self._th_lim_max) - self._relu(th-self._th_lim_min)
        dn = -self._relu(-th-self._th_lim_max)+self._relu(-th-self._th_lim_min)
        if (th > self._th_lim_min and thd > 0.0 or
                th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0
        return force

    def __call__(self, x, a):
        a_total = self._clip(a) + self._joint_lim_violation_force(x)
        return self._clip(a_total)


class LabeledBox(gym.spaces.Box):
    def __init__(self, labels, **kwargs):
        super(LabeledBox, self).__init__(**kwargs)
        assert len(labels) == self.high.size
        self.labels = labels


class GentlyTerminating(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            self.env.step(np.zeros(self.env.action_space.shape))
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()


class Timing:
    def __init__(self, fs, fs_ctrl):
        fs_ctrl_min = 50.0  # minimal control rate
        assert fs_ctrl >= fs_ctrl_min, \
            f"control frequency must be at least {fs_ctrl_min}"
        self.n_sim_per_ctrl = int(fs / fs_ctrl)
        assert fs == fs_ctrl * self.n_sim_per_ctrl, \
            "sampling frequency must be a multiple of the control frequency"
        self.dt = 1.0 / fs
        self.dt_ctrl = 1.0 / fs_ctrl
        self.render_rate = int(fs_ctrl)


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(self):
        # Gravity
        self.g = 9.81

        # Motor
        self.Rm = 8.4  # resistance
        self.kt = 0.042  # current-torque (N-m/A)
        self.km = 0.042  # back-emf constant (V-s/rad)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Jr = self.Mr * self.Lr ** 2 / 12  # inertia about COM (kg-m^2)
        self.Dr = 5e-6  # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.020  # mass (kg), original: 0.024
        self.Lp = 0.129  # length (m)
        self.Jp = self.Mp * self.Lp ** 2 / 12  # inertia about COM (kg-m^2)
        self.Dp = 1e-6  # viscous damping (N-m-s/rad), original: 0.0005

        # Constants for equations of motion
        self._c1 = self.Jr + self.Mp * self.Lr ** 2
        self._c2 = 0.25 * self.Mp * self.Lp ** 2
        self._c3 = 0.5 * self.Mp * self.Lp * self.Lr
        self._c4 = self.Jp + self._c2
        self._c5 = 0.5 * self.Mp * self.Lp * self.g

    def __call__(self, s, u):
        th, al, thd, ald = s
        voltage = u[0]

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c1 + self._c2 * np.sin(al) ** 2
        b = self._c3 * np.cos(al)
        c = self._c4
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c2 * np.sin(2 * al) * thd * ald \
             - self._c3 * np.sin(al) * ald * ald
        c1 = -0.5 * self._c2 * np.sin(2 * al) * thd * thd \
             + self._c5 * np.sin(al)
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
