import time
import numpy as np
import gym
from quanser_robots.cartpole.base import CartPoleDynamics
from scipy.linalg import solve_continuous_are, solve_discrete_are
class SwingupCtrl:
    """Rhythmically swinging metronome."""

    def __init__(self, dynamics=None, mu=3.):
        self.dynamics=CartPoleDynamics()
        self.mu=mu
        self.pd_control = False
        self.done = False
        self.render=False

        self.K = np.array([-50.,172.21,-48.,26.26])

    def __call__(self, state):
        x,sin_theta,cos_theta, x_dot, theta_dot = state

        theta = np.arctan2(sin_theta, cos_theta)

        dyna = self.dynamics
        Mp = self.dynamics._mp
        pl = self.dynamics._pl
        Jp = pl**2 * Mp /2.

        Ek = Jp/2. * theta_dot**2

        # since we use theta zero in the rest position, we have -theta dot and
        if -cos_theta > 0.995 or self.pd_control:
            self.render=True
            if theta > 0:
                alpha = theta-np.pi
            else:
                alpha = np.pi + theta
            print( 'alpha ', alpha, theta_dot)
            print('x', x, x_dot)
            u = np.matmul(self.K, (-np.array([x,alpha,x_dot,theta_dot])))

        else:
            u = -self.mu * Ek * np.sign(theta_dot * cos_theta)
        Vm = (dyna._Jeq * dyna._Rm * dyna._r_mp*u)/(dyna._eta_g * dyna._Kg * dyna._eta_m * dyna._Kt)\
              + dyna._Kg * dyna._Km * x_dot / dyna._r_mp
        Vm = np.clip(Vm,-7,7)
        return [Vm]


def main():
    env = gym.make('Cartpole-v0')

    ctrl = SwingupCtrl()
    obs = env.reset()
    i= 0
    while not ctrl.done:
        i+=1
        if i%20:
            env.render()
            i=0
        act = ctrl(obs)
        obs, _, _, _ = env.step(act)

    env.close()


if __name__ == "__main__":
    main()
