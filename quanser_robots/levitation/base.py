import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox, Timing

np.set_printoptions(precision=6, suppress=True)


class LevitationBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(LevitationBase, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        act_max = np.array([25.0])
        state_min, state_max = np.array([0.0, -5.0]), np.array([3.0, 5.0])
        sens_min, sens_max = np.array([0.0, -5.0]), np.array([3.0, 5.0])
        obs_min, obs_max = np.array([0.0, -5.0]), np.array([3.0, 5.0])

        # Spaces
        self.state_space = LabeledBox(
            labels=('ic', 'dic'),
            low=sens_min, high=state_max, dtype=np.float32)
        self.sensor_space = LabeledBox(
            labels=('ic', 'dic'),
            low=sens_min, high=sens_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('ic', 'dic'),
            low=obs_min, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('vc',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        self.goal = np.array([1.5])

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        # TODO: Make sure sending float64 is OK with real robot interface
        return self._sim_step([0.0])

    def _sim_step(self, a):
        """
        Update internal state of simulation and return an estimate thereof.

        :param a: action
        :return: state
        """
        raise NotImplementedError

    def _ctrl_step(self, a):
        x = self._state
        a_cmd = None
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = np.clip(a, self.action_space.low, self.action_space.high)
            x = self._sim_step(a_cmd)
        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, x, a):
        cost = (x[0] - self.goal).T @ (x[0] - self.goal) + 1e-3 * a[0] ** 2
        rwd = np.exp(-cost) * self.timing.dt_ctrl
        return np.float32(rwd), False

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        self._state = np.clip(self._state, self.state_space.low, self.state_space.high)
        obs = self._state
        return obs, rwd, done, {'s': self._state, 'a': act}

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

        # I/V Tf
        self.Tc = self.Lc / (self.Rc + self.Rs)
        self.Kp = self.Rc + self.Rs

    def __call__(self, s, u):
        ic, ic_des = s
        v = u

        A = - 1.0 / self.Tc
        B = self.Kp / self.Tc

        dic = A * ic + B * v[0]
        return dic
