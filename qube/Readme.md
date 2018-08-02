QUBE Interface
==============

Interface for controlling Qube (Furuta Pendulum).


Package contents
----------------
1) `doc.pdf` some modeling documentation from Quanser
1) `min.py` minimal communication interface -- just the step function
1) `qube.py` complete communication interface with examples
1) `sim.py` simulation and visualization


Run swing-up demo
-----------------
Make sure you have `python3` and `scipy` installed.

1) Run

        quarc_run -r Desktop\servers\qube\quarc_py_bridge_qube.rt-win64

   on the Windows PC to start Qube server.

1) Execute

        python3 qube.py

   on your PC to connect to Qube and run the swing-up demo.

1) After you are finished with experiments, run

       quarc_run -q Desktop\servers\qube\quarc_py_bridge_qube.rt-win64
   
   on the Windows PC to shut down Qube server.

By default, the Windows PC is expected to have the IP `192.172.162.1`.
You have to be in the same local network.


Use in your code
----------------
In your own code, use Qube as an OpenAI Gym environment
```python
from qube import Qube
qube = Qube()
x = qube.reset()
for _ in range(int(qube.fs)):
    u = [0.2]
    x = qube.step(u)
qube.close()
```
For further examples, see [qube.py](qube.py).

A physical model of QUBE and an explanation of the energy-based
swing-up controller are provided in [doc.pdf](doc.pdf).

