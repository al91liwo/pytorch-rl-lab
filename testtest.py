import quanser_robots
import gym

env = gym.make("CartpoleStabShort-v0")

print(env.observation_space.low)