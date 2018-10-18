import numpy as np
from ..common import VelocityFilter
from .base import LevitationBase, LevitationDynamics


class Levitation(LevitationBase):
    def __init__(self, fs, fs_ctrl):
        super(Levitation, self).__init__(fs, fs_ctrl)
        self.dyn = LevitationDynamics()
        self._sim_state = None

    def _calibrate(self):
        self._vel_filt = VelocityFilter(self.state_space.shape[0] - 1, dt=self.timing.dt)
        self._sim_state = np.zeros(self.state_space.shape)
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        ds = self.dyn(self._sim_state, a)

        # Update internal simulation state
        self._sim_state[1] = ds
        self._sim_state[0] += self.timing.dt * self._sim_state[1]

        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[0]
        vel = self._vel_filt(np.array([pos]))
        return np.concatenate([np.array([pos]), vel])

    def reset(self):
        self._calibrate()
        return self.step([0.0])[0]
