from gym.envs.registration import register


register(
    id='BallBalancerSim-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_sim:BallBalancerSim',
    max_episode_steps=500,
    kwargs={'fs': 100.0, 'fs_ctrl': 50.0, 'simplified_dyn': False}
)

register(
    id='BallBalancerSimSimpleDyn-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_sim:BallBalancerSim',
    max_episode_steps=500,
    kwargs={'fs': 100.0, 'fs_ctrl': 50.0, 'simplified_dyn': True}
)

register(
    id='BallBalancerRR-v0',
    entry_point='quanser_robots.ball_balancer.ball_balancer_rr:BallBalancerRR',
    max_episode_steps=5000,
    kwargs={'ip': '130.83.164.52', 'fs_ctrl': 50.0}
)
