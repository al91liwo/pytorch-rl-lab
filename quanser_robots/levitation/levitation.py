import numpy as np
from ..common import VelocityFilter
from .base import CoilBase, CoilDynamics
from .base import LevitationBase, LevitationDynamics
from .ctrl import CurrentPICtrl


class Coil(CoilBase):
    def __init__(self, fs, fs_ctrl):
        super(Coil, self).__init__(fs, fs_ctrl)
        self.dyn = CoilDynamics()

    def _sim_step(self, s, a):
        self._sim_state = np.copy(s)

        sd = self.dyn(s, a)

        # Update internal simulation state
        self._sim_state[1] = sd
        self._sim_state[0] += self.timing.dt * self._sim_state[1]

        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[0]

        vel = self._sim_state[1]
        return np.concatenate([np.array([pos]), np.array([vel])])

        # vel = self._vel_filt(np.array([pos]))
        # return np.concatenate([np.array([pos]), vel])

    def reset(self):
        self._vel_filt = VelocityFilter(self.state_space.shape[0] - 1, dt=self.timing.dt)
        self._state = np.zeros(self.state_space.shape)
        return self.observe()


class Levitation(LevitationBase):
    def __init__(self, fs, fs_ctrl):
        super(Levitation, self).__init__(fs, fs_ctrl)
        self.dyn = LevitationDynamics()

        self.coil = Coil(fs, fs_ctrl)
        self.coil.reset()

        self.pictl = CurrentPICtrl(dt=fs_ctrl)

    def _sim_step(self, s, a):
        self._sim_state = np.copy(s)

        # apply reference to PI
        vc = self.pictl(self.coil._state, a)

        # apply PI action to coil
        self.coil.step(vc)

        # apply current to levitation
        xbdd = self.dyn(self._sim_state, self.coil._state)

        # Update internal simulation state
        self._sim_state[1] += self.timing.dt * xbdd
        self._sim_state[0] += self.timing.dt * self._sim_state[1]

        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[0]

        vel = self._sim_state[1]
        return np.concatenate([np.array([pos]), np.array([vel])])

        # vel = self._vel_filt(np.array([pos]))
        # return np.concatenate([np.array([pos]), vel])

    def reset(self):
        self._vel_filt = VelocityFilter(self.state_space.shape[0] - 1, dt=self.timing.dt)
        self._state = np.hstack((self.xb0, 0.0))
        return self.observe()
