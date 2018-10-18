import numpy as np
from ..common import VelocityFilter
from .base import DoublePendulumBase


class DoublePendulum(DoublePendulumBase):
    def __init__(self, fs, fs_ctrl):
        super(DoublePendulum, self).__init__(fs, fs_ctrl)
        self._sim_state = None

        # ToDo => Samuele ....
        self._dyn = None

    def _calibrate(self):
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0], dt=self.timing.dt)
        self._sim_state = 0.01 * np.float32(self._np_random.randn(self.state_space.shape[0]))
        self._state = self._zero_sim_step()

    def _sim_step(self, a):
        # Add a bit of noise to action for robustness
        a_noisy = a + 1e-6 * np.float32(self._np_random.randn(self.action_space.shape[0]))

        # Compute dynamics:
        # ToDo => Samuele ....

        # Update internal simulation state
        # ToDo => Samuele ....

        # Pretend to only observe position and obtain velocity by filtering
        # ToDo => Samuele ....

        return s_t

    def reset(self):
        self._calibrate()
        return self.step([0.0])[0]
