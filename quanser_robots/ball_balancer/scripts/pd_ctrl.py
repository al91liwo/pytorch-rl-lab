import gym

from quanser_robots.ball_balancer.ctrl import PDCtrl


if __name__ == "__main__":
    env = gym.make('BallBalancerSim-v0')
    ctrl = PDCtrl(kp=[7., 7.], kd=[0., 0.])
    obs, done = env.reset()

    while not done:
        env.render()
        act = ctrl(obs)
        obs, _, _, done = env.step(act)

    env.close()
