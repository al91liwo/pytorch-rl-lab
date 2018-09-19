import numpy as np
from common import QSocket, VelocityFilter
from qube.base import QubeBase, CalibrCtrl


class Qube(QubeBase):
    def __init__(self, ip, fs_ctrl):
        super(Qube, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._qsoc = QSocket(ip, x_len=2, u_len=1)
        self._vel_filt = None
        self._th_mid = None
        self._alpha_mid = None

    def _calibrate(self):
        # Reset calibration
        self._th_mid = 0.0
        self._alpha_mid = 0.0
        self._state = np.zeros(self.state_space.shape[0])

        # Record alpha offset if alpha == k * 2pi (happens upon reconnect)
        x = self._sim_step(None, [0.0])
        while np.abs(x[3]) > 1e-6:
            x = self._sim_step(None, [0.0])
        self._alpha_mid = x[1]

        # Find theta offset by going to joint limits
        act = CalibrCtrl()
        x = self._sim_step(None, [0.0])
        while not act.done:
            x = self._sim_step(None, [act(x)])
        self._th_mid = (act.go_right.th_lim + act.go_left.th_lim) / 2

        # Set current state
        self._state = self._sim_step(None, [0.0])

    def _sim_step(self, _, a):
        _, pos = self._qsoc.snd_rcv(a)
        pos[0] -= self._th_mid
        pos[1] -= self._alpha_mid
        self._vel_filt(pos)
        return np.r_[pos, self._vel_filt(pos)]

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._vel_filt = VelocityFilter(2)
        self._calibrate()
        return self.step(0.0)[0]

    def render(self, mode='human'):
        pass

    def close(self):
        self._sim_step(None, [0.0])
        self._qsoc.close()
