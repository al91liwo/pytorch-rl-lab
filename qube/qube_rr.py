import numpy as np
from common import QSocket, VelocityFilter
from .base import QubeBase
from .ctrl import CalibrCtrl


class Qube(QubeBase):
    def __init__(self, ip, fs_ctrl):
        super(Qube, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0],
                             u_len=self.action_space.shape[0])
        self._vel_filt = None
        self._th_mid = None
        self._alpha_mid = None

    def _calibrate(self):
        # Reset calibration
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0])
        self._th_mid = 0.0
        self._alpha_mid = 0.0
        self._state = np.zeros(self.state_space.shape[0])

        # Record alpha offset if alpha == k * 2pi (happens upon reconnect)
        x = self._zero_sim_step()
        while np.abs(x[3]) > 1e-5:
            x = self._zero_sim_step()
        self._alpha_mid = x[1]

        # Find theta offset by going to joint limits
        act = CalibrCtrl()
        x = self._zero_sim_step()
        while not act.done:
            x = self._sim_step(None, [act(x)])
        self._th_mid = (act.go_right.th_lim + act.go_left.th_lim) / 2

        # Set current state
        self._state = self._zero_sim_step()

    def _zero_sim_step(self):
        return self._sim_step(None, np.r_[0.0])

    def _sim_step(self, _, a):
        _, pos = self._qsoc.snd_rcv(a)
        pos[0] -= self._th_mid
        pos[1] -= self._alpha_mid
        return np.r_[pos, self._vel_filt(pos)]

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._calibrate()
        return self.step(0.0)[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
