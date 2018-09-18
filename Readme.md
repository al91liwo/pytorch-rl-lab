Quanser Clients
===============

Simulated environments and real robot interfaces for a set of Quanser platforms.


Getting Started
---------------
Install all robot interfaces by executing

    pip3 install -e .

To confirm that the setup was successful, launch a Python3 console and run
    
    import gym
    import qube
    env = gym.make('Qube-v0')

If installation worked well, proceed to the robot-specific documentation

- [Qube](qube/Readme.md)
- [Ball Balancer](ball_balancer/Readme.md)


Developers and Maintainers
--------------------------
Awesome stuff developed by Boris Belousov, Fabio Muratore, and Hany Abdulsamad.
Add your name if you contributed so that we know whom to blame.
