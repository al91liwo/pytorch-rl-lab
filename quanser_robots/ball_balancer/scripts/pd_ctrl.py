import gym

from quanser_robots.ball_balancer.ctrl import PDCtrl


if __name__ == "__main__":
    # env = gym.make('BallBalancerSim-v0')
    env = gym.make('BallBalancerSimSimpleDyn-v0')
    ctrl = PDCtrl(kp=[14., 14.], kd=[0., 0.])
    obs, done = env.reset()

    while not done:
        env.render()
        act = ctrl(obs)
        obs, _, done, _ = env.step(act)

    env.close()
