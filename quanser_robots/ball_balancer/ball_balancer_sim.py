import numpy as np
from .base import BallBalancerBase, BallBalancerDynamics


class BallBalancerSim(BallBalancerBase):
    """
    Quanser 2 DoF Ball Balancer simulator class.
    """
    def __init__(self, fs, fs_ctrl):
        super().__init__(fs, fs_ctrl)
        self._dyn = BallBalancerDynamics(dt=self.timing.dt)

    def reset(self):
        """
        Reset the simulator
        :return: observation and false done flag
        """
        super().reset()
        obs, _, done, _ = self.step(np.zeros(self.action_space.shape))
        return obs, done

    def step(self, action):
        # Add a bit of noise to action for robustness
        # action = action + 1e-6 * np.float32(self._np_random.randn(self.action_space.shape[0]))

        # Apply action limits
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        accs, plate_angvel = self._dyn(self._state, self._plate_angs, action_clipped)

        # Integration step (symplectic Euler)
        self._state[4:] += accs * self.timing.dt  # next velocity
        self._state[:4] += self._state[4:] * self.timing.dt  # next position

        # Integration step (forward Euler, since we get the plate's angular velocities from the kinematics)
        self._plate_angs += plate_angvel * self.timing.dt  # next position

        # Also use the velocity filter in simulation (only the positions are measurable)
        pos = self._state[:4]
        vel = self._vel_filt(pos)
        obs = np.concatenate([pos, vel])

        reward = self._rew_fcn(obs, action)
        done = self._is_done()
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        super().render(mode)

    def close(self):
        pass
