from gym.envs.registration import register

register(
    id='DoublePendulum-v0',
    entry_point='quanser_robots.double_pendulum.double_pendulum:DoublePendulum',
    max_episode_steps=500,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0}
)

register(
    id='DoublePendulumRR-v0',
    entry_point='quanser_robots.double_pendulum.double_pendulum_rr:DoublePendulum',
    max_episode_steps=500,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 500.0}
)
