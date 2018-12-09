Levitation environment
================

Simulation and control environment for the Quanser Levitation.

To see Levitation in action, start the simulated current control demo

    python3 quanser_robots/levitation/examples/clsc_ctl.py


Package contents
----------------
1. `model.pdf` physical model description
2. `base.py` common functionality
3. `levitation.py` simulated environment
4. `levitation.py` real robot environment
5. `examples` example scripts


Controlling the real robot
--------------------------
To control the real robot, you must be in the same local network
with the control PC of the Levitation.
The easiest way is to connect directly via an Ethernet cable.
By default, the control PC has the IP address `192.172.162.1`.
The IP address can be changed in `quanser_robots/levitation/__init__.py`.

To run the swing-up demo on the real robot, perform the following steps:

1. Start the control server on the control PC

        quarc_run -r Desktop\servers\levitation\quarc_py_bridge_levitation.rt-win64

2. Launch the client application on your machine

        python3 quanser_robots/levitation/examples/clsc_ctl.py

3. At the end of the day, shut down the control server

       quarc_run -q Desktop\servers\levitation\quarc_py_bridge_levitation.rt-win64


### Control loop
The canonical way of using the real robot environment:
    
    import gym
    from quanser_robots import GentlyTerminating
    env = GentlyTerminating(gym.make('LevitationRR-v0'))
    ctrl = ...  # some function f: s -> a
    obs = env.reset()
    done = False
    while not done:
        act = ctrl(obs)
        obs, rwd, done, info = env.step(act)

Pay attention to the following important points:

- Reset the environment `env.reset()` right before running `env.step(act)`
  in a loop. If you forget to reset the environment and then send an action
  after some time of inactivity, you will get an outdated observation.

- Wrap the environment in the `GentlyTerminating` wrapper to ensure that
  a zero command is sent to the robot after an episode is finished.
  Levitation always keeps executing the last command it received, which may damage
  the motor if constant voltage is applied for too long.
