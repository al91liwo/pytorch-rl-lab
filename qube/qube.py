"""
QUBE Interface
==============

Class Qube is designed to resemble an OpenAI Gym environment.

This file provides

  1) class Qube for communicating with the robot

  2) several basic controllers

    - GoToLim goes to joint limits

    - Calibr finds the zero posture

    - Metronome performs a sinusoidal movement

    - PD does what you would expect

    - EnergyPump pumps in energy into the system

    - SwingUp switches from EnergyCtrl to PDCtrl for swing up

  3) some examples in comments and in __main__

"""

import numpy as np
import time
from common import QSocket, SymmetricBoxSpace, VelocityFilter


class Qube:
    """
    QUBE communication and control interface.


    Main functions
    ==============
    Call `qube = Qube()` to init communication
    and `qube.close()` to terminate it.

    Call `x = qube.reset()` to drive QUBE to zero position.

    Call `x = qube.step(u)` to send command `u` and receive state `x`.


    States and actions
    ==================
    State `x = (th, alpha, th_dot, alpha_dot)` consists of 2 angles in radians
    and their velocities. Angle `th` is the horizontal angle (rotary arm)
    limited to (-2.3, 2.3); angle `alpha` is the vertical angle (pendulum)
    with values in the range (-inf, inf). Zero position `(th, alpha) = (0, 0)`
    corresponds to the pendulum hanging down in the middle of the range of `th`.

    Control `u = Vm` is the motor voltage. Although there are no hardware
    restrictions on Vm, it is clipped before being sent to QUBE.
    You can increase the limit, but be careful since the rotary arm may detach
    from QUBE due to high momentum (it is attached by a magnet, similar to
    MacBook's power cord, therefore it safely breaks away at high speeds).


    Examples
    ========

    Basic PD control
    ----------------
        from qube import Qube
        qube = Qube()
        th_des = 1.0
        x = qube.reset()
        for _ in range(int(3 * qube.fs)):
          u = th_des - x[0]
          x = qube.step(u)
        qube.close()

    Running a controller
    --------------------
        from qube import Qube, MetronomeCtrl
        qube = Qube()
        qube.run(MetronomeCtrl())
        qube.close()


    Notes
    =====
      - If you are using QUBE interactively (e.g., running commands one by one
        from a Python console), be aware that you might be getting outdated
        sensor readings when you issue command `x = step(u)` after some time
        of inactivity. To remedy the situation, you can run `step(0.0)` several
        times to free up the receiving buffer before starting your actual
        controller (see `qube.run(act)` function for an example).

      - QUBE keeps executing the same command till you send a new one.
        Therefore, in order to prolong the lifetime of the motor,
        it is advisable to send `qube.step(0.0)` after your
        controller is finished but you want to stay connected to QUBE
        (e.g., when using it interactively from a Python console).

    """

    fs = 500.0   # Sampling frequency
    g = 9.81     # Gravity

    # Motor
    Rm = 8.4     # resistance
    kt = 0.042   # current-torque (N-m/A)
    km = 0.042   # back-emf constant (V-s/rad)

    # Rotary arm
    Mr = 0.095   # mass (kg)
    Lr = 0.085   # length (m)
    Jr = Mr * Lr ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dr = 0.0015  # equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum link
    Mp = 0.024   # mass (kg)
    Lp = 0.129   # length (m)
    Jp = Mp * Lp ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dp = 0.0005  # equivalent viscous damping coefficient (N-m-s/rad)

    def __init__(self, ip="192.172.162.1"):
        self.measurement_space = SymmetricBoxSpace(
            bound=np.array([2.3, np.inf]),
            labels=('theta', 'alpha')
        )
        self.state_space = SymmetricBoxSpace(
            bound=np.array([2.3, np.inf, np.inf, np.inf]),
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot')
        )
        self.action_space = SymmetricBoxSpace(
            bound=np.array([5.0]),
            labels=('motor_voltage',)
        )
        self._filt = VelocityFilter(2)
        self._qsoc = QSocket(ip, self.measurement_space.dim,
                            self.action_space.dim)
        print("Connected")
        self._th_mid = self._calibrate()

    def __call__(self, a):
        """Send command `a` and return `(t, s')`."""
        t, x = self._qsoc.snd_rcv(self.action_space.project(a))
        x[0] -= self._th_mid
        s = np.r_[x, self._filt(x)]
        return t, s

    def _calibrate(self):
        """Go to joint limits to find zero."""
        print('Calibrating...')
        self._th_mid = 0.0
        act = CalibrCtrl()
        self.run(act)
        th_mid = (act.go_right.th_lim + act.go_left.th_lim) / 2
        print('Calibration done')
        return th_mid

    def step(self, a):
        """Send command `a` and return the next state."""
        _, s = self.__call__(a)
        return s

    def run(self, act):
        """Run controller act till act.done == True."""
        x = self.step(np.array([0.0]))
        while not act.done:
            x = self.step(act(x))
        return x

    def reset(self):
        return self.run(PDCtrl())

    def close(self):
        self.step(np.array([0.0]))
        self._qsoc.close()
        print("Disconnected")


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
        if not self.done and np.sqrt(
                (th_des - x[0]) ** 2 + all_but_th_squared) < tol:
            self.done = True
        elif th_des and np.sqrt(all_but_th_squared) < tol / 5.0:
            # Increase P-gain on `th` when struggling to reach `th_des`
            K[0] += 0.01 * K[0]
        return [K[0] * (th_des - x[0]) - K[1] * x[1] - K[2] * x[2] - K[3] * x[3]]


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
        return [u]


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, positive=True):
        self.done = False
        self.th_lim = 0.0
        self.thd_max = 1e-4
        self.sign = 1 if positive else -1
        self.u_max = 0.8

    def __call__(self, x):
        if self.sign * self.th_lim < self.sign * x[0]:
            self.th_lim = x[0]
        if self.th_lim and np.abs(x[2]) < self.thd_max:
            self.done = True
        return [self.sign * self.u_max]


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
        return [u]


class EnergyCtrl:
    """Nonlinear energy shaping controller for a swing up."""

    def __init__(self, mu=50.0, Er=0.025, a_max=5.0):
        self.mu = mu  # P-gain on the energy (m/s/J)
        self.Er = Er  # reference energy (J)
        self.a_max = a_max  # maximum acceleration of the pendulum pivot (m/s^2)

    def __call__(self, x):
        alpha, alpha_dot = x[1], x[3]
        Ek = 0.5 * Qube.Jp * alpha_dot ** 2
        Ep = 0.5 * Qube.Mp * Qube.g * Qube.Lp * (1 - np.cos(alpha))
        E = Ek + Ep
        a = np.clip(
            self.mu * (self.Er - E) * np.sign(alpha_dot * np.cos(alpha)),
            -self.a_max, self.a_max)
        trq = Qube.Mr * Qube.Lr * a
        voltage = -Qube.Rm / Qube.kt * trq
        return [voltage]


class HybridCtrl:
    """Switch between PDCtrl and EnergyCtrl depending on the angle alpha."""

    def __init__(self, alpha_max_deg=20.0,
                 pd_ctrl=PDCtrl(K=[-2.0, 35.0, -1.5, 3.0]),
                 en_ctrl=EnergyCtrl()):
        self.alpha_max_deg = alpha_max_deg
        self.pd_ctrl = pd_ctrl
        self.en_ctrl = en_ctrl

    def __call__(self, x):
        # Limit alpha to [-pi, pi) with alpha = 0.0 at the top
        alpha = x[1] % (2 * np.pi) - np.pi
        # Small initial kick to get going
        if np.all(x == 0.0):
            return [0.01]
        # Engage PD when |alpha| < alpha_max_deg
        elif np.abs(alpha) < np.deg2rad(self.alpha_max_deg):
            y = x.copy()
            y[1] = alpha
            return self.pd_ctrl(y)
        # Pump up energy when |alpha| > alpha_max_deg
        else:
            return self.en_ctrl(x)


if __name__ == "__main__":
    # Start QUBE
    qube = Qube()

    # Convenience variables
    x_dim, u_dim = qube.state_space.dim, qube.action_space.dim
    x_labels = qube.state_space.labels
    u_labels = qube.action_space.labels

    # Prepare data storage
    n_cycles = int(7 * qube.fs)
    x_all = np.zeros((n_cycles, x_dim))
    u_all = np.zeros((n_cycles, u_dim))

    # Control loop
    print("Swinging up...")
    act = HybridCtrl()
    x = qube.reset()
    for i in range(n_cycles):
        u = act(x)
        x = qube.step(u)
        x_all[i] = x
        u_all[i] = u
    qube.step([0.0])
    print("Swing-up done")

    # Stop QUBE
    qube.close()

    # Plot
    try:
        import matplotlib.pyplot as plt

        plt.style.use('seaborn')
        fig, axes = plt.subplots(x_dim + u_dim, 1, figsize=(5, 8),
                                 tight_layout=True)
        legend_loc = 'lower right'
        t = np.linspace(0, n_cycles / qube.fs, n_cycles)
        for i in range(x_dim):
            axes[i].plot(t, x_all.T[i], label=x_labels[i], c=f'C{i}')
            axes[i].legend(loc=legend_loc)
        for i in range(u_dim):
            axes[x_dim+i].plot(t, u_all.T[i], label=u_labels[i], c=f'C{x_dim}')
            axes[x_dim+i].legend(loc=legend_loc)
        axes[0].set_ylabel('ang pos [rad]')
        axes[1].set_ylabel('ang pos [rad]')
        axes[2].set_ylabel('ang vel [rad/s]')
        axes[3].set_ylabel('ang vel [rad/s]')
        axes[4].set_ylabel('current [V]')
        axes[4].set_xlabel('time [samples]')
        plt.show()
    except ImportError:
        pass

