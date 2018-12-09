from gym.envs.registration import register
from .ctrl import SwingUpCtrl
from .base import Parameterized

register(
    id='Qube-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=300,
    kwargs={'fs': 100.0, 'fs_ctrl': 50.0}
)

register(
    id='QubeRR-v0',
    entry_point='quanser_robots.qube.qube_rr:Qube',
    max_episode_steps=300,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 50.0}
)
