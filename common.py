import socket
import struct
import numpy as np
from scipy import signal


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
        self._soc = socket.socket()
        self._soc.connect((ip, self._port))

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
        x = np.array(q[1:])
        return t, x


class SymmetricBoxSpace:
    """
    Generic real box space with symmetric boundaries.
    """
    def __init__(self, bound, labels):
        assert isinstance(bound, np.ndarray)
        assert bound.size == len(labels)
        self.bound_lo = -bound
        self.bound_up = bound
        self.dim = self.bound_lo.size
        self.labels = labels

    def project(self, ele):
        assert isinstance(ele, np.ndarray)
        return np.clip(ele, self.bound_lo, self.bound_up)


class VelocityFilter:
    """
    Create discrete velocity filter from continuous one.
    """
    def __init__(self, x_len, num=(50, 0), den=(1, 50), dt=0.002):
        """
        Initialize discrete filter coefficients
        :param x_len: number of measured state variables to receive
        :param num: continuous-time filter numerator
        :param den: continuous-time filter denominator
        :param dt: sampling time interval
        """
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel()
        self.a = derivative_filter[1]
        self.z = np.zeros((1, x_len))

    def __call__(self, x):
        xd, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return xd.ravel()
