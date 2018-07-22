import socket
import struct
import numpy as np


class QSocket:
    """
    Class for communication with Matlab.
    """
    def __init__(self, ip, x_len, u_len):
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
        :return: t, x: timestamp, state vector
        """
        self._soc.send(struct.pack(self._u_fmt, u))
        data = self._soc.recv(self._buf_size)
        q = struct.unpack(self._x_fmt, data)
        t = q[0]
        x = np.array(q[1:])
        return t, x


if __name__ == "__main__":
    ip = "130.83.164.122"
    q_sock = QSocket(ip, 2, 1)
    t, x = q_sock.snd_rcv(0.0)
    print(t, x)
