import numpy as np
import time
from ..common import QSocket, VelocityFilter
from .base import CartpoleBase
from .ctrl import GoToLimCtrl, PDCtrl


class Cartpole(CartpoleBase):
    def __init__(self, ip, fs_ctrl):
        super(Cartpole, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)

        # Initialize Socket:
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0], u_len=self.action_space.shape[0])

        # Save the relative limits:
        self._calibrated = False
        self._x_lim = np.zeros(2, dtype=np.float32)

    def _calibrate(self, verbose=False):
        if self._calibrated:
            return

        if verbose:
            print("\n\nCalibrate Cartpole:")

        # Retrieve current state:
        sensor = self._qsoc.snd_rcv(np.array([0.0]))

        # Reset calibration
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0], x_init=sensor)
        self._sens_offset = np.zeros(self.sensor_space.shape[0], dtype=np.float32)

        # Go to the left:
        if verbose:
            print("\tGo to the Left:\t\t\t", end="")

        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=True)

        while not ctrl.done :
            a = ctrl(state)
            state = self._sim_step(a)

        if ctrl.success:
            self._x_lim[1] = state[0]
            if verbose: print("\u2713")
        else:
            if verbose: print("\u274C ")
            raise RuntimeError("Going to the left limit failed.")

        # Go to the right
        if verbose: print("\tGo to the Right:\t\t", end="")
        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=False)

        while not ctrl.done :
            a = ctrl(state)
            state = self._sim_step(a)

        if ctrl.success:
            self._x_lim[0] = state[0]
            if verbose: print("\u2713")
        else:
            if verbose: print("\u274C ")
            raise RuntimeError("Going to the right limit failed.")

        # Activate the absolute cart position:
        self._calibrated = True

    def _center_cart(self, verbose=False):

        # Center the cart:
        if verbose:
            print("\tCentering the Cart:\t\t\t", end="")
        s_des = np.array([+0.0, +0.0, +0.0, +0.0])

        t_max = 100.0
        ctrl = PDCtrl(s_des=s_des)
        state = self._zero_sim_step()

        t0 = time.time()
        while not ctrl.done and (time.time() - t0) < t_max:
            a = ctrl(state)
            state = self._sim_step(a)

        # Set the current state:
        self._state = self._zero_sim_step()
        if not ctrl.done:
            if verbose: print("\u274C ")
            raise RuntimeError("Centering of the cart failed. Error: {0:.2e} > {1:.2e}".format(np.sum((state - s_des)**2), ctrl.tol))

        elif verbose:
            print("\u2713")

    def _sim_step(self, a):
        pos = self._qsoc.snd_rcv(a)

        # Transform the relative cart position to [-0.4, +0.4]
        if self._calibrated:
            pos[0] = (pos[0] - self._x_lim[0]) - 1./2. * (self._x_lim[1] - self._x_lim[0])

        return np.concatenate([pos, self._vel_filt(pos)])

    def reset(self):

        # Reconnect to the system:
        self._qsoc.close()
        self._qsoc.open()

        # The system only needs to be calibrated once, as this is a bit time consuming:
        self._calibrate()

        # Center the cart in the middle @ x = 0.0
        self._center_cart()
        return self.step([0.0])[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
