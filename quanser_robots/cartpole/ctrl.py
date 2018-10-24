import numpy as np
import time
from .cartpole import CartPoleDynamics


class PDCtrl:
    """
    PD controller for the cartpole environment.
    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, s_des=np.zeros(4), tol=5e-4):
        self.K = K if K is not None else np.array([20.0, 0.0, 0.0, 0.0])

        self.done = False
        self.s_des = s_des
        self.tol = tol

    def __call__(self, s):

        # Compute the voltage:
        err = self.s_des - s
        v = np.dot(self.K.transpose(), err)

        # Check for completion:
        if np.sum(err**2) <= self.tol:
            self.done = True

        return np.array([v], dtype=np.float32)

class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, s_init, positive=True):
        self.done = False
        self.success = False
        self.x_init = s_init[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1

        self.sign = 1 if positive else -1
        self.u_max = self.sign * np.array([1.5])

        self._t_init = False
        self._t0 = 0.0
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, s):
        x, _, xd, _ = s

        # Initialize the time:
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

        # Compute voltage:
        if (time.time() - self._t0) < self._t_min:
            u = self.u_max

        elif np.abs(xd) < self.xd_max: # and np.abs(x - self.x_init) > self.delta_x_min:
            u = np.zeros(1)
            self.success = True
            self.done = True

        elif (time.time() - self._t0) > self._t_max:
            u = np.zeros(1)
            self.success = False
            self.done = True

        else:
            u = self.u_max

        return u

def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta

class SwingupCtrl:
    """Swing up and balancing controller"""

    def __init__(self, long=False, mu=18.):
        self.dynamics = CartPoleDynamics(long=long)
        self.mu=mu
        self.pd_control = False
        self.pd_activated = False
        self.done = False

        self.K = np.array([-41.833, 173.4362, -46.1359, 16.2679])

    def __call__(self, state):
        x,sin_theta,cos_theta, x_dot, theta_dot = state

        alpha, theta = get_angles(sin_theta,cos_theta)

        dyna = self.dynamics
        Mp = self.dynamics._mp
        pl = self.dynamics._pl
        Jp = (pl)**2 * Mp /3.

        Ek = Jp/2. * theta_dot**2
        Ep = Mp*dyna._g*pl*(1-np.cos(theta))
        Er = 2*Mp*dyna._g*pl # ==0

        # since we use theta zero in the rest position, we have -theta dot and
        if np.abs(alpha) < 0.1745 or self.pd_control:
            if not self.pd_activated:
                print("PD Control")
            self.pd_activated = True

            u = np.matmul(self.K, (-np.array([x,alpha,x_dot,theta_dot])))
        else:
            umax = 18
            u = np.clip(self.mu * (Ek+Ep-Er) * np.sign(theta_dot * np.cos(theta)),-umax, umax)
            if self.pd_activated:
                print("PD Control Terminated")
                self.done = True
                self.pd_activated = False

        Vm = (dyna._Jeq * dyna._Rm * dyna._r_mp*u)/(dyna._eta_g * dyna._Kg * dyna._eta_m * dyna._Kt)\
              + dyna._Kg * dyna._Km * x_dot / dyna._r_mp
        Vm = np.clip(Vm,-24,24)

        return [Vm, u]


class SwingdownCtrl:
    """Swing down and keep in center and resting position."""

    def __init__(self, long=False, mu=14., epsilon=1E-4):
        self.dynamics = CartPoleDynamics(long=long)
        self.mu = mu
        self.pd_control = False
        self.pd_activated = False
        self.done = False
        self.epsilon = epsilon

        self.K = np.array([0., 0.1, 0.1, 0.1])

    def __call__(self, state):
        x,sin_theta,cos_theta, x_dot, theta_dot = state

        alpha, theta = get_angles(sin_theta,cos_theta)

        dyna = self.dynamics
        Mp = self.dynamics._mp
        pl = self.dynamics._pl
        Jp = (pl)**2 * Mp /3.

        Ek = Jp/2. * theta_dot**2
        Ep = Mp*dyna._g*pl*(1-np.cos(theta))
        Er = 2*Mp*dyna._g*pl # ==0

        # since we use theta zero in the rest position, we have -theta dot and
        if np.abs(theta) < 0.025 or self.pd_control:
            if not self.pd_activated:
                print("PD")
                self.pd_activated = True

            u = np.matmul(self.K, (-np.array([x,theta,x_dot,theta_dot])))
        else:
            umax = 10
            mu = self.mu * np.sqrt(np.abs(np.clip(theta_dot,-1,1)))
            u = -np.clip(mu * (Ek+Ep-Er) * np.sign(theta_dot * np.cos(theta)),-umax, umax)
            if self.pd_activated:
                print("energy")
                self.done = True
                self.pd_activated = False

        error = np.mean(np.square(np.array([x,theta,x_dot,theta_dot])))
        if error < self.epsilon:
            print("Resting position")
            self.done = True

        Vm = (dyna._Jeq * dyna._Rm * dyna._r_mp*u)/(dyna._eta_g * dyna._Kg * dyna._eta_m * dyna._Kt)\
              + dyna._Kg * dyna._Km * x_dot / dyna._r_mp
        Vm = np.clip(Vm,-24,24)

        return [Vm, u]