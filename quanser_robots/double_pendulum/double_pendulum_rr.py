import numpy as np
from ..common import QSocket, VelocityFilter
from .base import DoublePendulumBase


class DoublePendulum(DoublePendulumBase):
    def __init__(self, ip, fs_ctrl):
        super(Cartpole, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)

        # Initialize Socket:
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0], u_len=self.action_space.shape[0])
        self._sens_offset = None

    def _calibrate(self):
        # Reset calibration
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0])
        self._sens_offset = np.zeros(self.sensor_space.shape[0], dtype=np.float32)

        # Reset the cartpole to the center:
        # ToDo => Michael .....

        # Record alpha offset if alpha == k * 2pi (happens upon reconnect)
        x = self._zero_sim_step()
        if np.abs(x[1]) > np.pi:
            self._sens_offset[1] = x[1]

        # Set current state
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        pos = self._qsoc.snd_rcv(a)
        pos -= self._sens_offset
        return np.concatenate([pos, self._vel_filt(pos)])

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._calibrate()
        return self.step([0.0])[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
