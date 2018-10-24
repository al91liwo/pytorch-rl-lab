from gym.envs.registration import register

register(
    id='Cartpole-v0',
    entry_point='quanser_robots.cartpole.cartpole:Cartpole',
    max_episode_steps=500,
    kwargs={'fs': 500.0, 'fs_ctrl': 500.0}
)

register(
    id='CartpoleRR-v0',
    entry_point='quanser_robots.cartpole.cartpole_rr:Cartpole',
    max_episode_steps=1000,
    kwargs={'ip': '130.83.164.56', 'fs_ctrl': 500.0}
    #kwargs={'ip': '192.161.0.5', 'fs_ctrl': 500.0}
)

