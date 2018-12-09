import numpy as np

from .base import BallBalancerBase, BallBalancerDynamics
from ..common import QSocket, VelocityFilter


class BallBalancerRR(BallBalancerBase):
    """
    Quanser 2 DoF Ball Balancer real robot class.
    """
    def __init__(self, fs_ctrl, ip="130.83.164.52", simplified_dyn=False):
        super().__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._dyn = BallBalancerDynamics(dt=self.timing.dt, simplified_dyn=simplified_dyn)

        # Initialize communication
        self._qsoc = QSocket(ip, self.sensor_space.shape[0], self.action_space.shape[0])

        self._tol = 0.  # disable tolerance for done flag

    def reset(self):
        super().reset()
        # Cancel and re-open the connection to the socket
        self._qsoc.close()
        self._qsoc.open()

        # Initialize velocity filter
        # Send actions and receive sensor measurements. One extra send & receive for initializing the filter.
        pos_meas = self._qsoc.snd_rcv(np.array([0.0, 0.0]))
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0],
                                        dt=self.timing.dt,
                                        x_init=pos_meas)

        # Start gently with a zero action
        obs, _, done, _ = self.step(np.array([0.0, 0.0]))

        return obs, done

    def step(self, action):
        """
        Send command and receive next state.
        """
        info = {'action_raw': action}
        # Apply action limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._curr_action = action

        # Send actions and receive sensor measurements
        pos_meas = self._qsoc.snd_rcv(action)

        # Construct the state from measurements and observer (filter)
        obs = np.r_[pos_meas, self._vel_filt(pos_meas)]
        self._state = obs

        reward = self._rew_fcn(obs, action)
        done = self._is_done()  # uses the state estimated from measurements

        self._step_count += 1
        return obs, reward, done, info

    def render(self, mode='human'):
        super().render(mode)

    def close(self):
        # Terminate gently with a zero action
        self.step(np.array([0.0, 0.0]))

        # Cancel the connection to the socket
        self._qsoc.close()


if __name__ == "__main__":
    bb = BallBalancerRR()
    s = bb.step(np.array([0.0, 0.0]))
    print("state: ", s)
