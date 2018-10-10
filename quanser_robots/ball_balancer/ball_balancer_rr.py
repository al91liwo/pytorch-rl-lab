import numpy as np
from ..common import QSocket
from .base import BallBalancerBase


class BallBalancerRR(BallBalancerBase):
    def __init__(self, fs_ctrl, ip="130.83.164.52"):
        # Call constructor of parent class
        super().__init__(fs=500.0, fs_ctrl=fs_ctrl)

        # Initialize communication
        self._qsoc = QSocket(ip, self.sensor_space.shape[0], self.action_space.shape[0])

    def reset(self):
        super().reset()
        # Cancel and re-open the connection to the socket
        self._qsoc.close()
        self._qsoc.open()

        # Start gently with a zero action
        obs, _, done, _ = self.step(np.array([0.0, 0.0]))

        return obs, done

    def step(self, action):
        """
        Send command and receive next state.
        """
        # Apply action limits
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._curr_action = action

        # Send actions and receive sensor measurements
        pos_meas = self._qsoc.snd_rcv(action)

        # Construct the state from measurements and observer (filter)
        state = np.r_[pos_meas, self._vel_filt(pos_meas)]

        self._step_count += 1
        return state

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
