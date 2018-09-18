import numpy as np
from common import QSocket, VelocityFilter
from qube.base import QubeBase, CalibrCtrl


class Qube(QubeBase):
    def __init__(self, ip):
        super(Qube, self).__init__()
        self._qsoc = QSocket(ip, x_len=2, u_len=1)
        self._vel_filt = None
        self._th_mid = None
        self._alpha_mid = None

    def _calibrate(self):
        # Reset calibration
        self._th_mid = 0.0
        self._alpha_mid = 0.0
        self._state = np.zeros(self.state_space.shape[0])

        # Register alpha offset if alpha == k * 2pi (happens upon reconnect)
        x, _ = self._snd_rcv(0.0)
        while np.abs(x[3]) > 1e-6:
            x, _ = self._snd_rcv(0.0)
        self._alpha_mid = x[1]

        # Find theta offset
        act = CalibrCtrl()
        x, _ = self._snd_rcv(0.0)
        while not act.done:
            x, _ = self._snd_rcv(act(x))
        self._th_mid = (act.go_right.th_lim + act.go_left.th_lim) / 2

        # Set current state
        self._state, _ = self._snd_rcv(0.0)

    def _snd_rcv(self, a):
        a_clip = np.clip(np.r_[a], -self.act_max, self.act_max)
        pos = None
        for _ in range(self.cmd_dur):
            _, pos = self._qsoc.snd_rcv(a_clip)
            pos[0] -= self._th_mid
            pos[1] -= self._alpha_mid
            self._vel_filt(pos)
        return np.r_[pos, self._vel_filt(pos)], a_clip

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._vel_filt = VelocityFilter(2)
        self._calibrate()
        return self.step(0.0)[0]

    def render(self, mode='human'):
        pass

    def close(self):
        self._snd_rcv(0.0)
        self._qsoc.close()
