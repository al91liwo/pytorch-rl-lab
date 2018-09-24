Quanser Robots
==============

Simulated environments and real robot interfaces for a set of Quanser platforms.


Getting Started
---------------
To install the package, execute

    pip3 install -e .

To confirm that the setup was successful, launch a Python3 console and run
    
    import gym
    import quanser_robots
    env = gym.make('Qube-v0')

If installation worked well, proceed to the robot-specific documentation

- [Qube](quanser_robots/qube/Readme.md)
- [Ball Balancer](quanser_robots/ball_balancer/Readme.md)


Developers and Maintainers
--------------------------
Awesome stuff developed by Boris Belousov, Fabio Muratore, and Hany Abdulsamad.
Add your name if you contributed, so that we know whom to blame.
