import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox, Timing

np.set_printoptions(precision=6, suppress=True)

import numpy as np
from quanser_robots.common import LabeledBox
from ..common import Base, Timing
np.set_printoptions(precision=6, suppress=True)


class DoublePendulumBase(Base):

    def __init__(self, fs, fs_ctrl):

        super(DoublePendulumBase, self).__init__(fs, fs_ctrl)
        self._state = None
        self._vel_filt = None

        # Limits TODO: change limits
        self._x_lim = 0.814 / 2. #[m]

        act_max = np.array([24.0])
        # TODO: correct
        state_max = np.array([0.814 / 2., np.inf, np.inf, np.inf, np.inf, np.inf])
        sens_max = np.array([np.inf, np.inf, np.inf])
        obs_max = np.array([0.814 / 2. , 1.0,
                            1.0, np.inf, np.inf, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('x', 'theta1', 'theta2'),
            low=-sens_max, high=sens_max, dtype=np.float32)
        self.state_space = LabeledBox(
            labels=('x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('x', 'theta1', 'theta2', 'x_dot', 'theta1_dot', 'theta2_dot'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Function to ensure that state and action constraints are satisfied:
        self._lim_act = ActionLimiter()


        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _rwd(self, x, a):
        # TODO: change
        _, th, _, _, _, _ = x
        rwd = -np.cos(th)
        return np.float32(rwd), False

    def _observation(self, state):
        """
        A observation is provided given the internal state.

        :param state: (x, theta, x_dot, theta_dot)
        :type state: np.array
        :return: (x, sin(theta), cos(theta), x_dot, theta_dot)
        :rtype: np.array
        """
        return state


class ActionLimiter:

    def __init__(self):
        pass

    def _joint_lim_violation_force(self, x):
        pass

    def __call__(self, x, a):
        return a






