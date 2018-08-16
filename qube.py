import numpy as np

from quanser_clients.common import QSocket, SymmetricBoxSpace, VelocityFilter


class Qube:
    def __init__(self, ip="130.83.164.122"):
        # theta - arm angle; alpha - pendulum angle; (0, 0) = down position
        self.measurement_space = SymmetricBoxSpace(
            bound=np.array([2.3, np.inf]),
            labels=('theta', 'alpha')
        )
        # s = [theta, alpha, theta_dot, alpha_dot]
        self.state_space = SymmetricBoxSpace(
            bound=np.array([2.3, np.inf, np.inf, np.inf]),
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot')
        )
        # a = motor voltage
        self.action_space = SymmetricBoxSpace(
            bound=np.array([5.0]),
            labels=('motor_voltage',)
        )
        # Initialize velocity filter
        self.vel_filt = VelocityFilter(self.measurement_space.dim)

        # Initialize communication
        self._soc = QSocket(ip, self.measurement_space.dim, self.action_space.dim)

    def step(self, a):
        """Send command and receive next state."""
        t, x = self._soc.snd_rcv(self.action_space.project(a))
        s = np.r_[x, self.vel_filt(x)]
        return t, s


if __name__ == "__main__":
    qube = Qube()
    t, s = qube.step(np.array([0.0]))
    print(t, s)
