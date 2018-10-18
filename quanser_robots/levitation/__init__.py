from gym.envs.registration import register

register(
    id='Levitation-v0',
    entry_point='quanser_robots.levitation.levitation:Levitation',
    max_episode_steps=5000,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0}
)

register(
    id='LevitationRR-v0',
    entry_point='quanser_robots.levitation.levitation_rr:Levitation',
    max_episode_steps=500,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 500.0}
)
