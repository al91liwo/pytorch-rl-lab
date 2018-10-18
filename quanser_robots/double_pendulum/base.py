import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox, Timing

np.set_printoptions(precision=6, suppress=True)


class DoublePendulumBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(DoublePendulumBase, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        safety_th_lim = 1.5
        act_max = np.array([0.0])

        sens_min = np.array([0.0, 0.0, 0.0])
        sens_max = np.array([0.0, 0.0, 0.0])

        state_min = np.array([0.0, -np.inf, 0.0, -np.inf, 0.0, -np.inf])
        state_max = np.array([0.0, +np.inf, 0.0, +np.inf, 0.0, +np.inf])

        obs_min = np.array([0.0, -np.inf, 0.0, -np.inf, 0.0, -np.inf])
        obs_max = np.array([0.0, +np.inf, 0.0, +np.inf, 0.0, +np.inf])

        # Spaces
        # ToDo Should we add something additional to the observations?
        self.sensor_space = LabeledBox(
            labels=('pos', 'theta'),
            low=sens_min, high=sens_max, dtype=np.float32)

        self.state_space = LabeledBox(
            labels=('x', 'x_dot', 'theta0', 'theta0_dot', 'theta1', 'theta1_dot'),
            low=state_min, high=state_max, dtype=np.float32)

        self.observation_space = LabeledBox(
            labels=('x', 'x_dot', 'theta0', 'theta0_dot', 'theta1', 'theta1_dot'),
            low=obs_min, high=obs_max, dtype=np.float32)

        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)

        # ToDo How are we defining the reward?
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Function to ensure that state and action constraints are satisfied:
        # ToDo This has to be adapted!!
        self._lim_act = ActionLimiter(self.state_space, self.action_space, safety_th_lim)

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

        # Simulate n-steps:
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = self._lim_act(x, a)
            x = self._sim_step(a_cmd)

        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, x, a):
        # ToDo How do we define the reward function?

        # Qube Reward Function:
        # th, al, thd, ald = x
        # al_mod = al % (2 * np.pi) - np.pi
        # cost = al_mod**2 + 5e-3*ald**2 + 1e-1*th**2 + 2e-2*thd**2 + 3e-3*a[0]**2
        # done = not self.state_space.contains(x)
        # rwd = np.exp(-cost) * self.timing.dt_ctrl

        return np.float32(0.0), False

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):

        # Compute Reward:
        rwd, done = self._rwd(self._state, a)

        # Apply the action:
        self._state, act = self._ctrl_step(a)

        # Assemble State:
        obs = np.array(self._state, copy=True, dtype=np.float32)
        return obs, rwd, done, {'s': self._state, 'a': act}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):

        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = action_space.high[0] / (self._th_lim_max - self._th_lim_min)

        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)

    # ToDo What is this doing?? -> This has to be redone!
    def _joint_lim_violation_force(self, x):
        th, _, thd, _ = x

        up = self._relu(th-self._th_lim_max) - self._relu(th-self._th_lim_min)
        dn = -self._relu(-th-self._th_lim_max)+self._relu(-th-self._th_lim_min)

        if (th > self._th_lim_min and thd > 0.0 or th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0

        return force

    def __call__(self, x, a):
        force = self._joint_lim_violation_force(x)
        return self._clip(force if force else a)

