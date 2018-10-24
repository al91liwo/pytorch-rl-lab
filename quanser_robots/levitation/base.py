import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox, Timing

np.set_printoptions(precision=6, suppress=True)


class CoilBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(CoilBase, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        act_max = np.array([24.0])
        state_min, state_max = np.array([0.0, -1e3]), np.array([3.0, 1e3])
        sens_min, sens_max = np.array([0.0, -1e3]), np.array([3.0, 1e3])
        obs_min, obs_max = np.array([0.0, -1e3]), np.array([3.0, 1e3])

        # Spaces
        self.state_space = LabeledBox(
            labels=('ic', 'icd'),
            low=sens_min, high=state_max, dtype=np.float32)
        self.sensor_space = LabeledBox(
            labels=('ic', 'icd'),
            low=sens_min, high=sens_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('ic', 'icd'),
            low=obs_min, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('vc',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        self.icg = np.array([1.0]) # target current

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _sim_step(self, s, a):
        """
        Update internal state of simulation and return an estimate thereof.

        :param a: action
        :return: state
        """
        raise NotImplementedError

    def _ctrl_step(self, a):
        s = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = np.clip(a, self.action_space.low, self.action_space.high)
            s = self._sim_step(s, a_cmd)
            s = np.clip(s, self.state_space.low, self.state_space.high)
        return s, a_cmd  # return the last applied (clipped) command

    def _rwd(self, s, a):
        cost = (s[0] - self.icg).T @ (s[0] - self.icg) + 1e-3 * a[0] ** 2
        rwd = np.exp(-cost) * self.timing.dt_ctrl
        return np.float32(rwd), False

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        obs = self.observe()
        return obs, rwd, done, {'s': self._state, 'a': act}

    def observe(self):
        return self._state

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class CoilDynamics:
    """Solve equation V = (Rc + Rs) * I + Lc * dI """

    def __init__(self):
        # Coil
        self.Lc = 0.4125        # coil induction
        self.Nc = 2450          # coil loops
        self.lc = 0.0825        # coil length
        self.rc = 8e-3          # coil core radius
        self.Rc = 10.0          # coil resistance
        self.Km = 6.3508e-5     # magnetic force constant

        # Ball
        self.Mb = 0.068     # ball mass
        self.rb = 0.0127    # ball radius
        self.Tb = 0.014     # ball travel

        # Sensor
        self.Rs = 1.0              # sensor resistance
        self.KB = self.Tb / 5.0    # sensor sensitivity

        # I/V Tf
        self.Tc = self.Lc
        self.T0 = self.Rs + self.Rc
        self.Kc = 1.0

    def __call__(self, s, a):
        A = - self.T0 / self.Tc
        B = 1.0 / self.Tc
        return A * s[0] + B * a[0]



class LevitationBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(LevitationBase, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        act_max = np.array([3.0])
        state_min, state_max = np.array([0.0, -np.inf]), np.array([0.014, np.inf])
        sens_min, sens_max = np.array([0.0, -np.inf]), np.array([0.014, np.inf])
        obs_min, obs_max = np.array([0.0, -np.inf]), np.array([0.014, np.inf])

        # Spaces
        self.state_space = LabeledBox(
            labels=('xb', 'dxb'),
            low=sens_min, high=state_max, dtype=np.float32)
        self.sensor_space = LabeledBox(
            labels=('xb', 'dxb'),
            low=sens_min, high=sens_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('xb', 'dxb'),
            low=obs_min, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('ic',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        self.xb0 = np.array([0.014]) # operating point
        self.xbg = np.array([0.008]) # goal gap

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _sim_step(self, s, a):
        """
        Update internal state of simulation and return an estimate thereof.

        :param a: action
        :return: state
        """
        raise NotImplementedError

    def _ctrl_step(self, a):
        s = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = np.clip(a, self.action_space.low, self.action_space.high)
            s = self._sim_step(s, a_cmd)
            s = np.clip(s, self.state_space.low, self.state_space.high)
        return s, a_cmd  # return the last applied (clipped) command

    def _rwd(self, s, a):
        cost = (s[0] - self.xbg).T @ (s[0] - self.xbg) + 1e-3 * a[0] ** 2
        rwd = np.exp(-cost) * self.timing.dt_ctrl
        return np.float32(rwd), False

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        obs = self.observe()
        return obs, rwd, done, {'s': self._state, 'a': act}

    def observe(self):
        return self._state

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class LevitationDynamics:
    """Solve equation xdd = - (Km * ic^2) / (2 * Mb * xb^2) + g."""

    def __init__(self):
        # Nature
        self.g = 9.81                   # gravity
        self.mu0 = 4.0 * np.pi * 1e-7   # magnetic permeability constant

        # Coil
        self.Lc = 0.4125        # coil induction
        self.Nc = 2450          # coil loops
        self.lc = 0.0825        # coil length
        self.rc = 8e-3          # coil core radius
        self.Rc = 10.0          # coil resistance
        self.Km = 6.3508e-5     # magnetic force constant

        # Ball
        self.Mb = 0.068     # ball mass
        self.rb = 0.0127    # ball radius
        self.Tb = 0.014     # ball travel

        # Sensor
        self.Rs = 1.0              # sensor resistance
        self.KB = self.Tb / 5.0    # sensor sensitivity

    def __call__(self, s, a):
        return - self.Km * a[0] ** 2 / (2 * self.Mb * s[0] ** 2) + self.g

