from gym.envs.registration import register

register(
    id='Cartpole-v0',
    entry_point='quanser_robots.qube.qube:Qube',
    max_episode_steps=500,
    kwargs={'fs': 100.0, 'fs_ctrl': 50.0}
)

register(
    id='CartpoleRR-v0',
    entry_point='quanser_robots.cartpole.cartpole_rr:Cartpole',
    max_episode_steps=500,
    kwargs={'ip': '192.172.162.1', 'fs_ctrl': 50.0}
)
