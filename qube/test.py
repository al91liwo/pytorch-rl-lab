from qube import Qube
qube = Qube()
th_des = 1.0
x = qube.reset()
for _ in range(int(3 * qube.fs)):
    u = th_des - x[0]
    x = qube.step(u)
qube.close()
