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
import os
import ast


def parse_config(configfile):
    run_configs = []
    with open(configfile, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            run_configs.append(row)
    return run_configs


def write_config(config, configfile):
    with open(configfile, 'w+') as csvfile:
        w = csv.DictWriter(csvfile, config.keys(), delimiter=';')
        w.writeheader()
        w.writerow(config)


def train_and_evaluate(env, outdir, config):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("dev: ", dev, config)
    steps = ast.literal_eval(config["steps"])
    warmup_samples = ast.literal_eval(config["warmup_samples"])
    buffer_size = ast.literal_eval(config["buffer_size"])
    batch_size = ast.literal_eval(config["batch_size"])
    actor_lr = ast.literal_eval(config["actor_lr"])
    critic_lr = ast.literal_eval(config["critic_lr"])
    noise_decay = ast.literal_eval(config["noise_decay"])
    tau = ast.literal_eval(config["tau"])
    actor_hidden_layers = ast.literal_eval(config["actor_hidden_layers"])
    critic_hidden_layers = ast.literal_eval(config["critic_hidden_layers"])
    run_outdir = os.path.join(outdir, "{}_{}".format(config["run_id"], config["env"]))
    lr_decay = ast.literal_eval(config["lr_decay"])
    lr_min = ast.literal_eval(config["lr_min"])

    if not os.path.exists(run_outdir):
        os.makedirs(run_outdir)
    else:
        raise Exception("output directory '/{}' should not exist or be empty.".format(run_outdir))

    write_config(config, os.path.join(run_outdir, "parameters.csv"))

    # you need to test this
    print(steps, warmup_samples, buffer_size, batch_size, actor_lr, critic_lr, noise_decay, tau, actor_hidden_layers,
          critic_hidden_layers)

    ddpg = DDPG(env=env, dirname=run_outdir, steps=steps, warmup_samples=warmup_samples, buffer_size=buffer_size, batch_size=batch_size,
                actor_lr=actor_lr, critic_lr=critic_lr, lr_decay=lr_decay, lr_min=lr_min, noise_decay=noise_decay, action_space_limits=([-10.], [10.]),
                tau=tau, actor_hidden_layers=actor_hidden_layers, critic_hidden_layers=critic_hidden_layers, trial_horizon=2500,
                 device=dev)

    reward_record_training = ddpg.train()

    new_dirname = "{}_{}".format(run_outdir, reward_record_training)
    if os.path.exists(run_outdir):
        os.rename(run_outdir, new_dirname)
    if not os.path.exists(new_dirname):
        os.makedirs(new_dirname)
    run_outdir = new_dirname

    ddpg.actor_target.eval()
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
                state = torch.tensor(obs, dtype=torch.float32).to(dev).unsqueeze(0)

                action = ddpg.actor_target(state).squeeze().unsqueeze(0).cpu().detach().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            #                if step >= episodes - 1:
            #                   env.render()

            rew.append(total_reward)
    env.close()

    print(sum(rew) / len(rew))
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.plot(range(episodes), rew)
    plt.savefig(run_outdir + '/over100episodes.png')
    plt.clf()


def main():
    parser = ArgumentParser("DDPG")
    parser.add_argument('hyperparameters', type=str,
                        help='.csv file containing rows for each hyperparameter-set to test')
    parser.add_argument('outdir', type=str, help='Directory that contains the results of the runs')

    args = parser.parse_args()

    run_configs = parse_config(args.hyperparameters)

    for config in run_configs:
        try:
            env = gym.make(config['env'])

            train_and_evaluate(env, args.outdir, config)

            env.close()
        except Exception as e:
            print("error", e, "in config:", config)


main()
