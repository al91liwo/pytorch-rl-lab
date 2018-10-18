import gym

from quanser_robots.ball_balancer.ctrl import DummyCtrl


if __name__ == "__main__":
    env = gym.make('BallBalancerSim-v0')
    ctrl = DummyCtrl(env.action_space, duration=10)
    obs, done = env.reset()

    while not done:
        env.render()
        act = ctrl(obs)
        obs, _, done, _ = env.step(act)

    env.close()
