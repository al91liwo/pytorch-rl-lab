Double Pendulum
================

Simulation and control environment for the cartpole.


Package contents
----------------
1. [model.pdf](documentation/model.pdf) physical model description
2. [base.py](base.py) common functionality (here is defined the reward function)
3. [double_pendulum.py](double_pendulum.py) simulated environment
4. [double_pendulum_rr.py](double_pendulum_rr.py) real robot environment
5. `example\*` example scripts

Brief Introduction to the Cartpole
----------------------------------

Double pendulum is a classic environment used in control and in reinforcement learning. 
It consists in two poles linked together and with one extremity attached to a cart moving on a horizontal track.
The only actuated part of the system is the cart, 
which can be controlled usually by a horizontal force, or in our case by the voltage in 
input of the cart's engine. 

We propose a balancing tasks. A double pendulum swing-up results difficult to perform, also because our phisical system 
has hard limit on angle of the second pole, which cause high non-liniarities.

Our System
----------

Our Quanser platform, located in Room E303 S2|02 is the same used for the single cart-pole. 
It is possible to easily mount the double-pole on the cart. 

Observation of the state space
------------------------------

The overall system can be described by three entities: the position `x` of the cart on the track, the angle `theta1` of first pole w.r.t. the vertical upright position.
and the relative angle between the two poles `theta2` as shown in figure.
Please refer to [model.pdf](documentation/model.pdf) for the convention used. 
In order to control the system, we also need the derivates of this quantities, `x_dot`, `theta1_dot` and `theta2_dot`.


**Observation's state description:**

Position| 0 | 1         | 2         | 3     | 4          | 5          |
--------|---|-----------|-----------|-------|------------|------------|
Semantic| x |theta1     |theta2     | x_dot | theta1_dot | theta2_dot |


The Real System
----------------------------------

Every time we need to interact with the environment, we need to call the `reset` method.
The method will wait until the cartpole will be positioned in the center of the track and with the poles in upright position,
before passing and activating the controller subsequently.

*The pendulum needs to be positioned slowly and gently in its upright position.*

Control
--------------------------
To control the real robot, you must be in the same local network
with the control PC of the double pendulum.
The easiest way is to connect directly via an Ethernet cable.
By default, the control PC has the IP address `192.172.162.1`.
The IP address can be changed in `quanser_robots/cartpole/__init__.py`.

To run the demo on the real robot, perform the following steps:

1. Start the control server on the control PC

        quarc_run -r \......\quarc_py_bridge_????.rt-win64

2. Launch the client application on your machine

        python3 quanser_robots/qube/examples/first_steps.py

3. At the end of the day, shut down the control server

       quarc_run -q \.....\quarc_py_bridge_????.rt-win64


### Control loop
The canonical way of using the real robot environment:
    
    import gym
    from quanser_robots import GentlyTerminating
    env = GentlyTerminating(gym.make('CartpoleSwingRR-v0'))
    ctrl = ...  # some function f: s -> a
    obs = env.reset()
    done = False
    while not done:
        act = ctrl(obs)
        obs, rwd, done, info = env.step(act)

Pay attention to the following important points:

- Reset the environment `env.reset()` right before running `env.step(act)`
  in a loop. If you forget to reset the environment and then send an action
  after some time of inactivity, you will get an outdated observationm, and the system will be not calibrated.

- Wrap the environment in the `GentlyTerminating` wrapper to ensure that
  a zero command is sent to the robot after an episode is finished.
  Qube always keeps executing the last command it received, which may damage
  the motor if constant voltage is applied for too long.
