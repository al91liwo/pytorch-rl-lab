from common import QSocket, SymmetricBoxSpace, VelocityFilter
import numpy as np


class BallBalancer:
    def __init__(self, ip="130.83.164.122"):
        """
        Measurements:
        theta_x: plate angle in rad induced by the "X Axis Servo" (angle around the negative y axis)
        theta_y: plate angle in rad induced by the "Y Axis Servo" (angle around the negative x axis)
        pos_x: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        pos_y: ball position in meters along the x axis estimated by the "PGR Find Object" block from Quanser
        """
        # Initialize spaces for measurements, states, and actions
        self.measurement_space = SymmetricBoxSpace(
            bound=np.array([np.pi/4.0, np.pi/4.0, 0.15, 0.15]),
            labels=['theta_x', 'theta_y', 'pos_x', 'pos_x']
        )
        self.state_space = SymmetricBoxSpace(
            bound=np.array([np.pi/4.0, np.pi/4.0, 0.15, 0.15, np.inf, np.inf, np.inf, np.inf]),
            labels=['theta_x', 'theta_y', 'pos_x', 'pos_x', 'theta_x_dot', 'theta_y_dot', 'pos_x_dot', 'pos_x_dot']
        )
        self.action_space = SymmetricBoxSpace(
            bound=np.array([5.0, 5.0]),
            labels=('motor_V_x', 'motor_V_y')
        )

        # Initialize velocity filter
        self._filt = VelocityFilter(self.measurement_space.dim)

        # Initialize communication
        self._soc = QSocket(ip, self.measurement_space.dim, self.action_space.dim)

    def step(self, a):
        """
        Send command and receive next state.
        """
        t, x = self._soc.snd_rcv(self.action_space.project(a))
        s = np.r_[x, self._filt(x)]
        return t, s


if __name__ == "__main__":
    bb = BallBalancer()
    t, s = bb.step(np.array([0.0, 0.0]))
    print("t: ", t)
    print("state: ", s)
