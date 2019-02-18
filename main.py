import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG
from DDPG import batch_size_schedulers
import numpy as np
import numpy.random as rnd
from argparse import ArgumentParser
import csv


def parse_config(configfile):
	run_configs = []
	with open(configfile, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=';')
		for row in reader:
			run_configs.append(row)
	return run_configs


def train_and_evaluate(env, outdir, config):
	dev = "cuda" if torch.cuda.is_available() else "cpu"

	steps = config["steps"]
	warmup_samples = config["warmup_samples"]
	buffer_size = config["buffer_size"]
	batch_size = config["batch_size"]
	actor_lr = config["actor_lr"]
	critic_lr = config["critic_lr"]
	noise_decay = config["noise_decay"]
	tau = config["tau"]
	actor_hidden_layers = list(map(int, config["actor_hidden_layers"].split(',')))
	critic_hidden_layers= list(map(int, config["critic_hidden_layers"].split(',')))

	#you need to test this
	print(steps, warmup_samples, buffer_size, batch_size, actor_lr, critic_lr, noise_decay, tau, actor_hidden_layers, critic_hidden_layers)

	ddpg = DDPG(env=env, steps=steps, warmup_samples=warmup_samples, buffer_size=buffer_size, batch_size=batch_size,
			actor_lr=actor_lr, critic_lr=critic_lr, noise_decay=noise_decay, action_space_limits=([-5.], [5.]), tau=tau,
			actor_hidden_layers=actor_hidden_layers, critic_hidden_layers=critic_hidden_layers, dirname=outdir, device=dev)

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
	ddpg.save_model(result_dirname, config["env"])
	plt.xlabel("episodes")
	plt.ylabel("reward")
	plt.plot(range(episodes), rew)
	plt.savefig(result_dirname+'/over100episodes.png')
	plt.clf()

parser = ArgumentParser("DDPG")

def main():

	parser = ArgumentParser()
	parser.add_argument('hyperparameters', type=str, help='.csv file containing rows for each hyperparameter set to test')
	parser.add_argument('outdir', type=str, help='Directory that contains the results of the runs')

	args = parser.parse_args()

	run_configs = parse_config(args.hyperparameters)


	for config in run_configs:

		env = gym.make(config['env'])


		train_and_evaluate(env, args.outdir, config)

		env.close()

main()