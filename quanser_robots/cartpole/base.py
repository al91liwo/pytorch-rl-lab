import numpy as np
import gym
from gym.utils import seeding
from quanser_robots.common import LabeledBox
from scipy.linalg import solve_continuous_are
np.set_printoptions(precision=6, suppress=True)


class CartpoleBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        super(CartpoleBase, self).__init__()
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits TODO: change limits
        self._x_lim = 0.814 / 2. #[m]

        #safety_th_lim = 1.5
        act_max = np.array([5.0])
        state_max = np.array([0.814 / 2., np.inf, np.inf, np.inf])
        sens_max = np.array([np.inf, np.inf])
        obs_max = np.array([0.814 / 2. , 1.0,
                            1.0, np.inf, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('x', 'theta'),
            low=-sens_max, high=sens_max, dtype=np.float32)
        self.state_space = LabeledBox(
            labels=('x', 'theta', 'x_dot', 'theta_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('x',  'sin_th', 'cos_th', 'x_dot', 'th_dot'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Function to ensure that state and action constraints are satisfied
        self._lim_act = None #ActionLimiter(self.state_space,
                             #         self.action_space,
                             #         safety_th_lim)

        # ToDo @Samuele: How are we defining the reward?
        self.reward_range = (0.0, self.timing.dt_ctrl)
        safety_th_lim=None
        # Function to ensure that state and action constraints are satisfied:
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
        for _ in range(self.timing.n_sim_per_ctrl):
            a_cmd = a #self._lim_act(x, a)
            x = self._sim_step(a_cmd)
        return x, a_cmd  # return the last applied (clipped) command

    def _rwd(self, x, a):
        # TODO: change
        _, th, _, _ = x
        rwd = -np.cos(th)
        return np.float32(rwd), False

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def _observation(self, state):
        # x, sin(alpha), cos(alpha), x_dot, alpha_dot
        return state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]

    def step(self, a):
        rwd, done = self._rwd(self._state, a)
        self._state, act = self._ctrl_step(a)
        obs = self._observation(self._state)
        return obs, rwd, done, {'s': self._state, 'a': act}

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


class Timing:
    def __init__(self, fs, fs_ctrl):
        fs_ctrl_min = 50.0  # minimal control rate
        assert fs_ctrl >= fs_ctrl_min, \
            "control frequency must be at least {}".format(fs_ctrl_min)
        self.n_sim_per_ctrl = int(fs / fs_ctrl)
        assert fs == fs_ctrl * self.n_sim_per_ctrl, \
            "sampling frequency must be a multiple of the control frequency"
        self.dt = 1.0 / fs
        self.dt_ctrl = 1.0 / fs_ctrl
        self.render_rate = int(fs_ctrl)


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):
        """ I comment otherwise the code is not working
        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = \
            action_space.high[0] / (self._th_lim_max - self._th_lim_min)
        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)
        """
    """
    Michael, the state is encoded x, theta, x_dot, theta_dot
    I don't understand why do you want to have limit for theta actually
    """
    def _joint_lim_violation_force(self, x):
        """
        th, _, thd, _ = x
        up = self._relu(th-self._th_lim_max) - self._relu(th-self._th_lim_min)
        dn = -self._relu(-th-self._th_lim_max)+self._relu(-th-self._th_lim_min)
        if (th > self._th_lim_min and thd > 0.0 or
                th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0
        return force
        """
        return x

    def __call__(self, x, a):
        force = self._joint_lim_violation_force(x)
        return self._clip(force if force else a)

class PhysicSystem:

    def __init__(self, timing, **kwargs):
        self.timing = timing
        for k in kwargs:
            setattr(self, k, kwargs[k])
            #TODO: make possible also to define it
            setattr(self, k + "_dot", 0.)

    def add_acceleration(self, **kwargs):
        for k in kwargs:
            setattr(self, k + "_dot", getattr(self, k + "_dot") + self.timing.dt * kwargs[k])
            setattr(self, k, getattr(self, k) + self.timing.dt * getattr(self, k + "_dot"))

    def get_state(self, entities_list):
        ret = []
        for k in entities_list:
            ret.append(getattr(self, k))
        return np.array(ret)


class CartPoleDynamics:

    def __init__(self):

        self._eta_m = 1.            # Motor efficiency  []
        self._eta_g = 1.             # Planetary Gearbox Efficiency []
        self._Kg = 3.72             # Planetary Gearbox Gear Ratio
        self._Jm = 3.9E-7           # Rotor inertia [kg.m^2]
        self._r_mp = 6.35E-3        # Motor Pinion radius [m]
        self._Rm = 2.6              # Motor armature Resistence [Ohm]
        self._Kt = .00767           # Motor Torque Constant [N.zz/A]
        self._Km = .00767           # Motor Torque Constant [N.zz/A]

        self._mp = 0.127  # mass of the pole [kg]
        self._mc = 0.38  # mass of the cart [kg]
        self._pl = 0.3365 / 2.  # half of the pole lenght [m]

        self._Jp = self._pl**2 * self._mp   # Pole inertia [kg.m^2]
        self._Jeq = self._mc + (self._eta_g * self._Kg**2 * self._Jm)/(self._r_mp**2)
        self._JT = self._Jeq * self._Jp + self._mp * self._Jp + self._Jeq * self._mp * self._pl**2

        self._Beq = 5.4             # Equivalent Viscous damping Coefficient
        self._Bp = 0.0024           # Viscous coefficient at the pole

        self._x_lim = 0.814 / 2.  # limit of position of the cart [m]

        self._g = 9.81  # gravitational acceleration [m.s^-2]

        self.A = np.array(
            [[0.,       0.,     0.02,     0.],
             [0.,       0.,     0.,     0.02],
             [0., self._mp**2*self._pl**2*self._g, -(self._Jp +self._mp*self._pl**2)*self._Beq, -self._mp*self._pl*self._Bp],
             [0.,(self._Jeq+self._mp)*self._mp*self._pl*self._g, -self._mp*self._pl*self._Beq, -(self._Jeq+self._mp)*self._Bp]]
        )/self._JT


        self.B = np.array(
            [0.,
             0.,
             self._Jp+self._mp*self._pl**2,
             self._mp*self._pl
             ]
        )/self._JT

    def __call__(self, s, V_m):
        x, theta, x_dot, theta_dot = s

        F = (self._eta_g*self._Kg*self._eta_m*self._Kt)/(self._Jeq*self._Rm*self._r_mp) *(-self._Kg*self._Km*x_dot/self._r_mp + self._eta_g*V_m)

        A = np.array([[np.cos(theta), self._pl],
                      [self._mp + self._mc, self._pl * self._mp * np.cos(theta)]])
        b = np.array([-self._g * np.sin(theta),
                      F  + self._mp * self._pl * theta_dot ** 2 * np.sin(theta)])

        return np.linalg.solve(A, b)




