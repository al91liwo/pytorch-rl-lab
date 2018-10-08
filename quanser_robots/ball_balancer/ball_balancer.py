import numpy as np
from .base import BallBalancerBase, BallBalancerDynamics


class BallBalancerSim(BallBalancerBase):
    def __init__(self, fs, fs_ctrl):
        super().__init__(fs, fs_ctrl)
        self._dyn = BallBalancerDynamics()
        self._state = None

    def reset(self):
        super().reset()
        return self.step([0.0])[0]

    def step(self, a):
        # Add a bit of noise to action for robustness
        a_noisy = a + 1e-6 * np.float32(self._np_random.randn(self.action_space.shape[0]))

        # Clip the action

        state_acc, plate_angvel = self._dyn(self._sim_state, a_noisy)

        # Update internal simulation state
        self._sim_state[3] += self.timing.dt * aldd
        self._sim_state[2] += self.timing.dt * thdd
        self._sim_state[1] += self.timing.dt * self._sim_state[3]
        self._sim_state[0] += self.timing.dt * self._sim_state[2]

        # Integration step (symplectic Euler)
        self.state[4:] += to.tensor([th_x_ddot, th_y_ddot, x_ddot, y_ddot]) * self._dt  # next velocity
        self.state[:4] += self.state[4:] * self._dt  # next position
        # Integration step (forward Euler, since we get the plate's angular velocities from the kinematics)
        self.plate_angs += to.tensor([a_dot, b_dot]) * self._dt  # next position

        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[:2]
        vel = self._vel_filt(pos)
        return np.concatenate([pos, vel])

    def render(self, mode='human'):
        super().render(mode)

    def close(self):
        pass
