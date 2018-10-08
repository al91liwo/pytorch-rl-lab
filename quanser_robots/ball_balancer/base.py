import numpy as np
import gym
from gym.utils import seeding

from ..common import VelocityFilter, LabeledBox

np.set_printoptions(precision=3, suppress=True)


class BallBalancerBase(gym.Env):
    def __init__(self, fs, fs_ctrl):
        """
        Base class for the Quanser 2 DoF Ball Balancer (simulation as well as real device)
        Measurements:
        theta_x: plate angle in rad induced by the X Axis Servo (angle around the negative y axis)
        theta_y: plate angle in rad induced by the Y Axis Servo (angle around the negative x axis)
        pos_x: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        pos_y: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        Action:
        V_x: voltage command for the X Axis Servo
        V_y: voltage command for the Y Axis Servo
        """
        super(BallBalancerBase, self).__init__()
        self._state = None
        self._step_count = None
        self.timing = Timing(fs, fs_ctrl)

        # Initialize spaces for measurements, states, and actions
        state_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, np.inf, np.inf, np.inf, np.inf])
        sens_max = state_max[:4]
        act_max = np.array([5.0, 5.0])

        self.sensor_space = LabeledBox(
            labels=('theta_x', 'theta_y', 'pos_x', 'pos_x'),
            low=-sens_max, high=sens_max, dtype=np.float32)
        self.state_space = LabeledBox(
            labels=('theta_x', 'theta_y', 'pos_x', 'pos_x', 'theta_x_dot', 'theta_y_dot', 'pos_x_dot', 'pos_x_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        # self.observation_space = LabeledBox(
        #     labels=('?', '?', '?', '?'),
        #     low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('V_x', 'V_y'),
            low=-act_max, high=act_max, dtype=np.float32)

        # Initialize velocity filter
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0])

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._step_count = 0
        self._state = np.zeros(self.state_space.shape)

    def step(self, a):
        raise NotImplementedError

    def render(self, mode='human'):
        """
        Cheap print to console
        """
        if mode == 'human':
            print("step: {:3}  |  in bounds: {:1}  |  state: {}".format(
                self._step_count, self.state_space.contains(self._state), self._state))
            # If the positions are out of bound
            if not self.state_space.contains(self._state):
                np.set_printoptions(precision=3, suppress=True)
                print("min state : ", self.state_space.low)
                print("last state: ", self._state)
                print("max state : ", self.state_space.high)

    def close(self):
        raise NotImplementedError


class BallBalancerDynamics:
    """
    Modeling the dynamics equations for the Quanser 2 DoF Ball Balancer
    """
    def __init__(self, dt):
        # System variables
        self.dt = dt  # integration step size [s]
        self.g = 9.81  # gravity constant [m/s**2]
        self.m_ball = 0.003  # mass of the ball [kg]
        self.r_ball = 0.019625  # radius of the ball [m]
        self.l_plate = 0.275  # length of the (square) plate [m]
        self.r_arm = 0.0254  # distance between the servo output gear shaft and the coupled joint [m]
        self.K_g = 70  # gear ratio [-]
        self.eta_g = 0.9  # gearbox efficiency [-]
        self.J_l = 5.2822e-5  # moment of inertia of the load [kg*m**2]
        self.J_m = 4.6063e-7  # motor armature moment of inertia [kg*m**2]
        self.eta_m = 0.69  # motor efficiency [-]
        self.k_m = 0.0077  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
        self.R_m = 2.6  # motor armature resistance
        self.B_eq = 0.015  # equivalent viscous damping coefficient w.r.t. load [N*m*s/rad]
        self.c_frict = 0.05  # viscous friction coefficient [N*s/m]
        self.V_thold_x_pos = 0.365  # voltage required to move the X Servo in positive direction
        self.V_thold_x_neg = -0.145  # voltage required to move the X Servo in negative direction
        self.V_thold_y_pos = 0.325  # voltage required to move the Y Servo in positive direction
        self.V_thold_y_neg = -0.125  # voltage required to move the Y Servo in negative
        self.ang_offset_a = 0  # constant plate angle offset around the x axis [rad]
        self.ang_offset_b = 0  # constant plate angle offset around the y axis [rad]

        # Derived constants
        self.J_ball = 2. / 5 * self.m_ball * self.r_ball ** 2  # inertia of the ball [kg*m**2]
        self.J_eq = self.eta_g * self.K_g ** 2 * self.J_m + self.J_l  # equivalent moment of inertia [kg*m**2]
        self.c_kin = 2. * self.r_arm / self.l_plate  # coefficient for the rod-plate kinematic
        self.A_m = self.eta_g * self.K_g * self.eta_m * self.k_m / self.R_m
        self.B_eq_v = (self.eta_g * self.K_g ** 2 * self.eta_m * self.k_m ** 2 + self.B_eq * self.R_m) / self.R_m
        self.zeta = self.m_ball * self.r_ball ** 2 + self.J_ball  # combined moment of inertial for the ball

    def __call__(self, state, plate_angs, action, simplified_dyn=False):
        """
        Nonlinear Dynamics
        :param state: the state [servo_angpos_x, servo_angpos_y, ball_pos_x, ball_pos_y,
                             servo_angvel_x, servo_angvel_y, ball_vel_x, ball_vel_y]
        :param plate_angs: angular position of the plate (additional info)
                           Note: plate_angs is not necessary in case of simplified_dyn=True
        :param action: unbounded action (no clipping in this function)
        :param simplified_dyn: flags if a dynamics model without Coriolis forces and without friction should be used
        :return: 
        """
        # State
        th_x = state[0]  # angle of the x axis servo (load)
        th_y = state[1]  # angle of the y axis servo (load)
        x = state[2]  # ball position along the x axis
        y = state[3]  # ball position along the y axis
        th_x_dot = state[4]  # angular velocity of the x axis servo (load)
        th_y_dot = state[5]  # angular velocity of the y axis servo (load)
        x_dot = state[6]  # ball velocity along the x axis
        y_dot = state[7]  # ball velocity along the y axis

        # Servos' angular accelerations
        th_x_ddot = (self.A_m * action[0] - self.B_eq_v * th_x_dot) / self.J_eq
        th_y_ddot = (self.A_m * action[1] - self.B_eq_v * th_y_dot) / self.J_eq

        # Plate (not part of the state since it is a redundant information)
        a = plate_angs[0] + self.ang_offset_a  # plate'state angle around the y axis (alpha)
        b = plate_angs[1] + self.ang_offset_b  # plate'state angle around the x axis (beta)
        a_dot = self.c_kin * th_x_dot * np.cos(th_x) / np.cos(a)  # angular velocity of the plate around the y axis
        b_dot = self.c_kin * -th_y_dot * np.cos(-th_y) / np.cos(b)  # angular velocity of the plate around the x axis
        # Plate'state angular accelerations (unused for simplified_dyn = True)
        a_ddot = 1. / np.cos(a) * (
                self.c_kin * (th_x_ddot * np.cos(th_x) - th_x_dot ** 2 * np.sin(th_x)) + a_dot ** 2 * np.sin(a))
        b_ddot = 1. / np.cos(b) * (
                self.c_kin * (-th_y_ddot * np.cos(-th_y) - th_y_dot ** 2 * np.sin(-th_y)) + b_dot ** 2 * np.sin(b))

        # kinematics: sin(a) = c_kin * sin(th_x)
        if simplified_dyn:
            # Ball dynamic without friction and Coriolis forces
            x_ddot = self.c_kin * self.m_ball * self.g * self.r_ball ** 2 * np.sin(th_x) / self.zeta  # symm inertia
            y_ddot = self.c_kin * self.m_ball * self.g * self.r_ball ** 2 * np.sin(th_y) / self.zeta  # symm inertia
        else:
            # Ball dynamic with friction and Coriolis forces
            x_ddot = (- self.c_frict * x_dot * self.r_ball ** 2  # friction
                      - self.J_ball * self.r_ball * a_ddot  # plate influence (necessary?)
                      + self.m_ball * x * a_dot ** 2 * self.r_ball ** 2  # centripetal
                      + self.c_kin * self.m_ball * self.g * self.r_ball ** 2 * np.sin(th_x)  # gravity
                      ) / self.zeta
            y_ddot = (- self.c_frict * y_dot * self.r_ball ** 2  # friction
                      - self.J_ball * self.r_ball * b_ddot  # plate influence (necessary?)
                      + self.m_ball * x * b_dot ** 2 * self.r_ball ** 2  # centripetal
                      + self.c_kin * self.m_ball * self.g * self.r_ball ** 2 * np.sin(th_y)  # gravity
                      ) / self.zeta

        # Return the state accelerations / plate velocities and do the integration outside this function
        state_acc = np.array([th_x_ddot, th_y_ddot, x_ddot, y_ddot])
        plate_angvel = np.array([a_dot, b_dot])
        return state_acc, plate_angvel


class Timing:
    def __init__(self, fs, fs_ctrl):
        fs_ctrl_min = 50.0  # minimal control rate
        assert fs_ctrl >= fs_ctrl_min, "control frequency must be at least {}".format(fs_ctrl_min)
        self.n_sim_per_ctrl = int(fs / fs_ctrl)
        assert fs == fs_ctrl * self.n_sim_per_ctrl, "sampling frequency must be a multiple of the control frequency"
        self.dt = 1.0 / fs
        self.dt_ctrl = 1.0 / fs_ctrl
        self.render_rate = int(fs_ctrl)
