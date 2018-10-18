from gym.envs.registration import register
from .ctrl import SwingUpCtrl

register(
    id='Qube-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=5000,
    kwargs={'fs': 300.0, 'fs_ctrl': 150.0}
)

register(
    id='QubeRR-v0',
    entry_point='quanser_robots.qube.qube_rr:Qube',
    max_episode_steps=500,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 50.0}
)
