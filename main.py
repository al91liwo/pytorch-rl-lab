import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG
from DDPG import batch_size_schedulers
import numpy as np
import numpy.random as rnd
from argparse import ArgumentParser

env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)

dev = "cuda" if torch.cuda.is_available() else "cpu"

def main():

	ddpg = DDPG(env=env, steps=500000, warmup_samples=20000, buffer_size=100000, batch_size=256,
			actor_lr=1e-3, critic_lr=1e-2, actor_lr_decay=.995, critic_lr_decay=.995, noise_decay=0.99, epochs=1, batch_size_scheduler=0, action_space_limits=([-5.], [5.]),
			actor_hidden_layers=[100, 100, 50], critic_hidden_layers=[100, 100], device=dev)

	result_dirname = ddpg.train()
	ddpg.actor_target.eval()
	ddpg.load_model()
	episodes = 100
	rew = []
	with torch.no_grad():
		for step in range(episodes):
			done = False
			obs = env.reset()
			total_reward = 0
			print(step)
			while not done:
				obs = ddpg.transformObservation(obs)
				state = torch.tensor(obs, dtype=torch.float32).to(dev)

				action = ddpg.actor_target(state).cpu().detach().numpy()
				obs, reward, done, _ = env.step(action)
				total_reward += reward

#                if step >= episodes - 1:
#                   env.render()


			rew.append(total_reward)
	env.close()


	print(sum(rew)/len(rew))
	ddpg.save_model(result_dirname, env_name)
	plt.xlabel("episodes")
	plt.ylabel("reward")
	plt.plot(range(episodes), rew)
	plt.savefig(result_dirname+'/over100episodes.png')
	plt.clf()

parser = ArgumentParser("DDPG")

while True:
    main()
