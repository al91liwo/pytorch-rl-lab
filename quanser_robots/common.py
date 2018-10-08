import socket
import struct
import numpy as np
from scipy import signal
import gym
from gym import spaces


class QSocket:
    """
    Class for communication with Quarc.
    """
    def __init__(self, ip, x_len, u_len):
        """
        Prepare socket for communication.
        :param ip: IP address of the Windows PC
        :param x_len: number of measured state variables to receive
        :param u_len: number of control variables to send
        """
        self._x_fmt = '>' + (x_len + 1) * 'd'  # +1 is for the timestamp
        self._u_fmt = '>' + u_len * 'd'
        self._buf_size = (x_len + 1) * 8  # 8 bytes for each double
        self._port = 9095  # fixed in Simulink model
        self._ip = ip
        self._soc = None

    def snd_rcv(self, u):
        """
        Send u and receive (t, x).
        :param u: control vector
        :return: t, x: timestamp, vector of measured states
        """
        self._soc.send(struct.pack(self._u_fmt, *u))
        data = self._soc.recv(self._buf_size)
        q = struct.unpack(self._x_fmt, data)
        t = q[0]
        x = np.array(q[1:], dtype=np.float32)
        return t, x

    def open(self):
        if self._soc is None:
            self._soc = socket.socket()
            self._soc.connect((self._ip, self._port))

    def close(self):
        if self._soc is not None:
            self._soc.close()
            self._soc = None


class SymmetricBoxSpace:
    """
    Generic real-valued box space with symmetric boundaries.
    """
    def __init__(self, bound: np.ndarray, labels: tuple):
        self.bound_lo = -bound
        self.bound_up = bound
        self.labels = labels
        self.dim = len(labels)

    def project(self, ele: np.ndarray):
        return np.clip(ele, self.bound_lo, self.bound_up)


class VelocityFilter:
    """
    Discrete velocity filter derived from a continuous one.
    """
    def __init__(self, x_len, num=(50, 0), den=(1, 50), dt=0.002, x_init=None):
        """
        Initialize discrete filter coefficients.
        :param x_len: number of measured state variables to receive
        :param num: continuous-time filter numerator
        :param den: continuous-time filter denominator
        :param dt: sampling time interval
        :param x_init: initial observation of the signal to filter
        """
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        if x_init is None:
            self.z = np.zeros((1, x_len), dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init):
        """
        This method can be used to set the initial state of the velocity filter.
        This is useful when the initial (position) observation
        has been retrieved and it is non-zero.
        Otherwise the filter would assume a very high velocity.
        :param x_init: initial observation
        """
        assert isinstance(x_init, np.ndarray)
        # Get the initial condition of the filter
        zi = signal.lfilter_zi(self.b, self.a)  # dim = order of the filter = 1
        # Set the filter state
        self.z = zi * x_init.reshape((1, -1))

    def __call__(self, x):
        xd, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return xd.ravel()


class LabeledBox(spaces.Box):
    """
    Adds `labels` field to gym.spaces.Box to keep track of variable names.
    """
    def __init__(self, labels, **kwargs):
        super(LabeledBox, self).__init__(**kwargs)
        assert len(labels) == self.high.size
        self.labels = labels


class GentlyTerminating(gym.Wrapper):
    """
    This env wrapper sends zero command to the robot when an episode is done.
    """
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            self.env.step(np.zeros(self.env.action_space.shape))
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()
