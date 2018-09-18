from gym.envs.registration import register

register(
    id='Qube-v0',
    entry_point='qube.qube:Qube',
    max_episode_steps=250
)

register(
    id='QubeRR-v0',
    entry_point='qube.qube_rr:Qube',
    max_episode_steps=250,
    kwargs={'ip': '192.172.162.1'}
)
