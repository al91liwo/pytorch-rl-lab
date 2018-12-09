Ball Balancer Environment
=========================

Simulation and control environment for the Quanser 2DoF Ball Balancer (ball-on-plate).


Package contents
----------------
1. `model.pdf` physical model description
2. `base.py` common functionality
3. `ball_balancer.py` simulated environment
4. `ball_balancer_rr.py` real robot environment
5. `ctrl.py` simple controllers
6. `examples` example scripts


Controlling the real robot
--------------------------
To control the real robot, you must be in the same local network with the control PC of the Ball Balancer.
By default, the control PC has the IP address `130.83.164.119` (you should double-check this).
The IP address passed to the BallBalancer's constuctor must match the one of the control PC.

To run the swing-up demo on the real robot, perform the following steps:

1. Start the control server on the control PC

        quarc_run -r Desktop\servers\qube\quarc_py_bridge_bb.rt-win64

2. Launch the client application on your machine

        python3 quanser_robots/qube/examples/<SCRIPT_NAME>.py

3. At the end of the day, shut down the control server

       quarc_run -q Desktop\servers\qube\quarc_py_bridge_bb.rt-win64


### Control loop
The canonical way of using the real robot environment:
    
    import gym
    from quanser_robots import GentlyTerminating
    env = GentlyTerminating(gym.make('BallBalancerRR-v0'))
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
  Qube always keeps executing the last command it received, which may damage
  the motor if constant voltage is applied for too long.
