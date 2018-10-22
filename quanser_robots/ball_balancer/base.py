import numpy as np
import gym
from gym.utils import seeding

from ..common import LabeledBox, Timing

np.set_printoptions(precision=3, suppress=True)


class BallBalancerBase(gym.Env):
    def __init__(self, fs, fs_ctrl, state_des=None, tol=5e-3):
        """
        Base class for the Quanser 2 DoF Ball Balancer (simulation as well as real device)
        Note: the information about the plate's angular position is not necessary for the simlified dynamics.
              Furthermore, it can be calculated from the inverse kinematics of the mechanism.
        Measurements:
        theta_x: x axis servo shaft angle
        theta_y: y axis servo shaft angle
        pos_x: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        pos_y: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        Auxiliary state info:
        alpha: plate's angle around the negative y axis (alpha)
        beta: plate's angle around the x axis (beta)
        Actions:
        V_x: voltage command for the X Axis Servo
        V_y: voltage command for the Y Axis Servo
        ---
        :param state_des: goal state
        :param tol: position tolerance [m]
        """
        super(BallBalancerBase, self).__init__()
        self._state = None
        self._vel_filt = None  # init in subclasses
        self._plate_angs = None  # auxiliary information about the plate's angular position
        self._dyn = None  # only necessary for BallBalancerSim, but might be beneficial for BallBalancerRR
        self.done = None
        self._step_count = 0
        self._curr_action = None  # only for plotting
        self.timing = Timing(fs, fs_ctrl)

        # Initialize spaces for measurements, states, and actions
        state_max = np.array([np.pi/4., np.pi/4., 0.15, 0.15, 4.*np.pi, 4.*np.pi, 0.5, 0.5])
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

        # Goal state, rewards and done flag
        self._state_des = np.zeros(self.state_space.shape) if state_des is None else state_des
        self._tol = tol
        self.Q = np.diag([1e-2, 1e-2, 1e-0, 1e-0, 1e-4, 1e-4, 1e-2, 1e-2])  # see dim of state space
        self.R = np.diag([1e-4, 1e-4])  # see dim of action space
        self.min_rew = 1e-4

        # Initialize random number generator
        self._np_random = None
        # self.seed()

        # Init storage for rendering
        self._anim_canvas = None
        self._anim_ball = None
        self._anim_plate = None

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reset the time and done flag
        self.done = False
        self._step_count = 0
        # Reset the state
        self._state = np.zeros(self.state_space.shape)
        self._plate_angs = np.zeros(2)

    def step(self, action):
        raise NotImplementedError

    def _rew_fcn(self, obs, action):
        err_s = (self._state_des - obs).reshape(-1,)  # or self._state
        err_a = action.reshape(-1,)
        quadr_cost = err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))

        obs_max = self.state_space.high.reshape(-1, )
        act_max = self.action_space.high.reshape(-1, )

        max_cost = obs_max.dot(self.Q.dot(obs_max)) + act_max.dot(self.R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        self.c_max = -1.0 * np.log(self.min_rew) / max_cost

        # Calculate the scaled exponential
        rew = np.exp(-self.c_max * quadr_cost)  # c_max > 0, quard_cost >= 0
        return float(rew)

    def _is_done(self):
        """
        Check if the state is out of bounds or if the goal is reached.
        :return: bool
        """
        # Calculate the Cartesian distance to the goal position (neglect other states)
        dist = np.linalg.norm(self._state_des[2:4] - self._state[2:4], ord=2)
        if dist <= self._tol or not self.state_space.contains(self._state):
            if dist <= self._tol:
                print("-- Done: reached goal position!")
                print("distance: ", dist)
            if not self.state_space.contains(self._state):
                print("-- Done: out of bounds!")
                print("min state : ", self.state_space.low)
                print("last state: ", self._state)
                print("max state : ", self.state_space.high)
            return True
        else:
            return False

    def render(self, mode, render_step=10):
        assert isinstance(self._dyn, BallBalancerDynamics), "Missing dynamics properties for simulation!"
        
        # Cheap printing to console
        if self._step_count % render_step == 0:
            print("time step: {:3}  |  in bounds: {:1}  |  state: {}  |  action: {}".format(
                self._step_count, self.state_space.contains(self._state), self._state, self._curr_action))

            if not self.state_space.contains(self._state):
                # State is out of bounds
                np.set_printoptions(precision=3)
                print("min state : ", self.state_space.low)
                print("last state: ", self._state)
                print("max state : ", self.state_space.high)

        # Render using vpython
        import vpython as vp
        vp.rate(30)
        d_plate = 0.01  # only for animation

        # Init render objects on first call
        if self._anim_canvas is None:
            self._anim_canvas = vp.canvas(width=800, height=600, title="Quanser Ball Balancer")
            self._anim_ball = vp.sphere(
                pos=vp.vector(self._state[2], self._state[3], self._dyn.r_ball),
                radius=self._dyn.r_ball,
                mass=self._dyn.m_ball,
                color=vp.color.red,
                canvas=self._anim_canvas,
            )
            self._anim_plate = vp.box(
                pos=vp.vector(0, 0, 0),
                size=vp.vector(self._dyn.l_plate, self._dyn.l_plate, d_plate),
                color=vp.color.green,
                canvas=self._anim_canvas,
            )
        #  Compute plate orientation
        a = self._plate_angs[0]  # plate's angle around the y axis (alpha)
        b = self._plate_angs[1]  # plate's angle around the x axis (beta)

        # Axis runs along the x direction
        self._anim_plate.axis = vp.vec(
            vp.cos(a),
            0,
            vp.sin(a),
        ) * self._dyn.l_plate
        # Up runs along the y direction (vpython coordinate system is weird)
        self._anim_plate.up = vp.vec(
            0,
            vp.cos(b),
            vp.sin(b),
        )

        # Compute ball position
        x = self._state[2]  # ball position along the x axis
        y = self._state[3]  # ball position along the y axis

        self._anim_ball.pos = vp.vec(
            x * vp.cos(b),
            y * vp.cos(a),
            self._dyn.r_ball + x * vp.sin(b) + y * vp.sin(a) + vp.cos(a) * d_plate / 2.,
        )

        # Set caption text
        self._anim_canvas.caption = f"""
            Plate angles: {b * 180/np.pi :2.2f}, {a * 180/np.pi :2.2f}
            Ball position: {x :1.3f}, {y :1.3f}
            """

    def close(self):
        raise NotImplementedError


class BallBalancerDynamics:
    """
    Modeling the dynamics equations for the Quanser 2 DoF Ball Balancer
    """
    def __init__(self, dt, simplified_dyn):
        """
        :param dt: simulation time step
        :param simplified_dyn: flags if a dynamics model without Coriolis forces and without friction should be used
        """
        self.simplified_dyn = simplified_dyn

        # System variables
        self.dt = dt  # integration step size [s]
        self.g = 9.81  # gravity constant [m/s**2]
        self.m_ball = 0.003  # mass of the ball [kg]
        self.r_ball = 0.019625  # radius of the ball [m]
        self.l_plate = 0.275  # length of the (square) plate [m]
        self.r_arm = 0.0254  # distance between the servo output gear shaft and the coupled joint [m]
        self.K_g = 70  # gear ratio [-]
        self.eta_g = 0.9  # gearbox efficiency [-]
        self.eta_m = 0.69  # motor efficiency [-]
        self.J_l = 5.2822e-5  # moment of inertia of the load [kg*m**2]
        self.J_m = 4.6063e-7  # motor armature moment of inertia [kg*m**2]
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

    def __call__(self, state, plate_angs, action):
        """
        Nonlinear Dynamics
        :param state: the state [servo_angpos_x, servo_angpos_y, ball_pos_x, ball_pos_y,
                                 servo_angvel_x, servo_angvel_y, ball_vel_x, ball_vel_y]
        :param plate_angs: angular position of the plate (additional info)
                           Note: plate_angs is not necessary in case of simplified_dyn=True
        :param action: unbounded action (no clipping in this function)
        :return: accelerations of the servo shaft angles and the ball positions
        """
        # Apply a voltage dead zone (i.e., below a certain amplitude the system does not move)
        # A very simple model of static friction. Experimentally evaluated the voltage required to get the plate moving.
        if self.V_thold_x_neg <= action[0] <= self.V_thold_x_pos:
            action[0] = 0
        if self.V_thold_y_neg <= action[1] <= self.V_thold_y_pos:
            action[1] = 0

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
        a = plate_angs[0] + self.ang_offset_a  # plate's angle around the negative y axis (alpha)
        b = plate_angs[1] + self.ang_offset_b  # plate's angle around the x axis (beta)
        a_dot = self.c_kin * th_x_dot * np.cos(th_x) / np.cos(a)  # angular velocity of the plate around the y axis
        b_dot = self.c_kin * -th_y_dot * np.cos(-th_y) / np.cos(b)  # angular velocity of the plate around the x axis
        # Plate'state angular accelerations (unused for simplified_dyn = True)
        a_ddot = 1. / np.cos(a) * (
                self.c_kin * (th_x_ddot * np.cos(th_x) - th_x_dot ** 2 * np.sin(th_x)) + a_dot ** 2 * np.sin(a))
        b_ddot = 1. / np.cos(b) * (
                self.c_kin * (-th_y_ddot * np.cos(-th_y) - th_y_dot ** 2 * np.sin(-th_y)) + b_dot ** 2 * np.sin(b))

        # kinematics: sin(a) = c_kin * sin(th_x)
        if self.simplified_dyn:
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
                      + self.m_ball * y * b_dot ** 2 * self.r_ball ** 2  # centripetal
                      + self.c_kin * self.m_ball * self.g * self.r_ball ** 2 * np.sin(th_y)  # gravity
                      ) / self.zeta

        # Return the state accelerations / plate velocities and do the integration outside this function
        accs = np.array([th_x_ddot, th_y_ddot, x_ddot, y_ddot])
        plate_angvel = np.array([a_dot, b_dot])
        return accs, plate_angvel


class BallBalancerKinematics:
    """
    Calculates and visualizes the kinematics from the servo shaft angles (th_x, th_x) to the plate angles (a, b).
    """

    def __init__(self, qbb):
        """
        :param qbb: QBallBalancerEnv object
        """
        self.qbb = qbb

        self.r = self.qbb.domain_param['r_arm']
        self.l = self.qbb.domain_param['l_plate'] / 2.
        self.d = 0.10  # roughly measured

        # Visualization
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(-0.5 * self.r, 1.2 * (self.r + self.l))
        self.ax.set_ylim(-1.0 * self.d, 2 * self.d)
        self.ax.set_aspect('equal')
        self.line1, = self.ax.plot([0, 0], [0, 0], marker='o')
        self.line2, = self.ax.plot([0, 0], [0, 0], marker='o')
        self.line3, = self.ax.plot([0, 0], [0, 0], marker='o')

    def __call__(self, th):
        """

        :param th: x or y
        :return: plate angle al pha or beta
        """
        import torch as to

        if not isinstance(th, to.Tensor):
            th = to.tensor(th, dtype=to.float32)

        # Update the lengths, e.g. if the domain has been randomized
        self.r = self.qbb.domain_param['r_arm']
        self.l = self.qbb.domain_param['l_plate'] / 2.
        self.d = 0.10  # roughly measured

        tip = self.rod_tip(th)
        ang = self.plate_ang(tip)
        self.render(th, tip)
        return ang

    def rod_tip(self, th):
        """
        Get Cartesian coordinates of the rod tip for one servo.
        :param th: current value of the respective servo shaft angle
        :return tip: 2D position of the rod tip in the sagittal plane
        """
        import torch as to
        # Initial guess for the rod tip
        tip_init = [self.r, self.l]  # [x, y] in the sagittal plane
        tip = to.tensor(tip_init, requires_grad=True)

        optimizer = to.optim.SGD([tip], lr=0.01, momentum=0.9)

        for i in range(200):
            optimizer.zero_grad()
            loss = self._loss_fcn(tip, th)
            loss.backward()
            optimizer.step()

        return tip

    def _loss_fcn(self, tip, th):
        """
        Cost function for the optimization problem, which only consists of 2 constraints that should be fulfilled.
        :param tip:
        :param th:
        :return: the cost value
        """
        import torch as to

        # Formulate the constrained optimization problem as an unconstrained using the known segment lengths
        rod_len = to.sqrt((tip[0] - self.r * to.cos(th)) ** 2 + (tip[1] - self.r * to.sin(th)) ** 2)
        half_palte = to.sqrt((tip[0] - self.r - self.l) ** 2 + (tip[1] - self.d) ** 2)

        return (rod_len - self.d) ** 2 + (half_palte - self.l) ** 2

    def plate_ang(self, tip):
        """
        Compute plate angle (alpha or beta) from the rod tip position which has been calculated from servo shaft angle
        (th_x or th_y) before.
        :return tip: 2D position of the rod tip in the sagittal plane (from the optimizer)
        """
        import torch as to
        ang = np.pi / 2. - to.atan2(self.r + self.l - tip[0], tip[1] - self.d)
        return float(ang)

    def render(self, th, tip):
        """
        Visualize using pyplot
        :param th:
        :param tip:
        """
        A = [0, 0]
        B = [self.r * np.cos(th), self.r * np.sin(th)]
        C = [tip[0], tip[1]]
        D = [self.r + self.l, self.d]

        self.line1.set_data([A[0], B[0]], [A[1], B[1]])
        self.line2.set_data([B[0], C[0]], [B[1], C[1]])
        self.line3.set_data([C[0], D[0]], [C[1], D[1]])
