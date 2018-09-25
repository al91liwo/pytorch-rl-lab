Quanser Robots
==============

Simulated environments and real robot interfaces for a set of Quanser platforms.


Getting Started
---------------
To install the package, execute

    pip install -e .

To confirm that the setup was successful, launch a Python console and run
    
    import gym
    import quanser_robots
    env = gym.make('Qube-v0')
    env.reset()
    env.render()

If installation worked well, proceed to the robot-specific documentation

- [Qube](quanser_robots/qube/Readme.md)
- [Ball Balancer](quanser_robots/ball_balancer/Readme.md)

If you are getting errors during installation that you do not know how to fix,
check below whether the requirements are satisfied, and if necessary follow
the [detailed installation instructions](docs/Install.md).


Requirements
------------
The main requirement comes from the 3D graphics library `vpython` which
is used for rendering the environments. It requires Python >= 3.5.3
(preferably Pyton 3.6.5).
Note that the default version of Python on Ubuntu 16.04 is Python 3.5.2,
so visualization will not work with it.
You can still use the environments though, just don't call `env.render()`.


Developers and Maintainers
--------------------------
Awesome stuff developed by Boris Belousov, Fabio Muratore, and Hany Abdulsamad.
Add your name if you contributed, so that we know whom to blame.
