Qube environment
================

Simulation and control of Quanser Qube (a.k.a. Furuta Pendulum).
Tested with Python 3.6.5.


Package contents
----------------
1) `doc.pdf` physical model specification
2) `base.py` common functionality and controllers
3) `qube.py` simulated environment
4) `qube_rr.py` real robot environment
5) `examples` show how to use the environments


Simulation
----------
Start a simulated swing-up demo

    python3 qube/examples/swing-up.py


Controlling real robot
----------------------
To control the real robot, connect your computer to the same
local network to which the Windows control PC is connected,
e.g., by connecting directly to the Windows PC via an Ethernet cable.
By default, the Windows PC has the IP address `192.172.162.1`.
In case you need to use a different address,
adjust the file `qube/__init__.py` accordingly.


To run the swing-up demo on the real robot, perform the following steps:

1) Start the control server on the Windows PC

        quarc_run -r Desktop\servers\qube\quarc_py_bridge_qube.rt-win64

2) Launch the client application on your machine

        python3 qube/examples/swing-up_rr.py

3) At the end of the day, shut down the control server

       quarc_run -q Desktop\servers\qube\quarc_py_bridge_qube.rt-win64


### Canonical example of a control loop
Here is the canonical way of using the real robot environment

    ctrl = ...  # some (stochastic) function f: s -> a
    obs = env.reset()
    done = False
    while not done:
        act = ctrl(obs)
        obs, rwd, done, info = env.step(act)
    env.step(0.0)

Pay attention to the following important points:

- Always reset the environment `env.reset()` before running `env.step(act)`
  in a loop. If you forget to reset the environment and then send an action
  after some time of inactivity, you will get an outdated observation.
  
- Send a zero command `env.step(0.0)` at the end of the control loop.
  The robot will keep executing the last command it received.
  The motor of the robot may get damaged if a constant non-zero voltage
  gets applied for too long.
