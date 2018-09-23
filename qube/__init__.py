from .base import GentlyTerminating
from .ctrl import SwingUpCtrl

from gym.envs.registration import register

register(
    id='Qube-v0',
    entry_point='qube.qube:Qube',
    max_episode_steps=500,
    kwargs={'fs': 200.0, 'fs_ctrl': 50.0}
)

register(
    id='QubeRR-v0',
    entry_point='qube.qube_rr:Qube',
    max_episode_steps=500,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 50.0}
)
