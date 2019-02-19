import copy
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
import os
import datetime


batch_size_schedulers = [
    lambda bs, e: bs,
    lambda bs, e: bs + e
]

class DDPG:
    
    def __init__(self, env, dirname, action_space_limits, buffer_size=10000, batch_size=64, epochs=1,
                 gamma=.99, tau=1e-2, steps=100000, warmup_samples=1000, noise_decay=0.9,
                 transform=lambda x: x, actor_lr=1e-3, critic_lr=1e-3, actor_lr_decay=1., critic_lr_decay=1., trial_horizon=5000,
                 actor_hidden_layers=[10, 10, 10], critic_hidden_layers=[10, 10, 10], batch_size_scheduler=0, device="cpu"):
        self.device = device
        self.env = env
        self.env_low = torch.tensor(action_space_limits[0], device=self.device)
        self.env_high = torch.tensor(action_space_limits[1], device=self.device)
        print(self.env_low, self.env_high)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.started = datetime.datetime.now()
        self.dirname = dirname
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_lr_decay = actor_lr_decay
        self.critic_lr_decay = critic_lr_decay
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.warmup_samples = warmup_samples
        self.epochs = epochs

        self.actor_network = ActorNetwork([self.state_dim, *actor_hidden_layers, self.action_dim], torch.tensor(self.env_low, device=self.device), torch.tensor(self.env_high, device=self.device)).to(self.device)
        self.critic_network = CriticNetwork([self.state_dim + self.action_dim, *critic_hidden_layers, 1]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor_network).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_network).to(self.device)
        
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, actor_lr_decay)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, critic_lr_decay)
        self.loss = nn.MSELoss()
        self.noise_decay = torch.tensor(noise_decay, device=self.device)
        self.trial_horizon = trial_horizon
        
        self.replayBuffer = ReplayBuffer(buffer_size, self.device)
        self.batch_size = batch_size
        self.n_batches = warmup_samples

        self.gamma = torch.tensor(gamma, device=self.device)

        self.tau = torch.tensor(tau, device=self.device)
        self.total_steps = steps
        self.noise_torch = torch.distributions.normal.Normal(0, self.env_high[0])
        self.transformObservation = transform

        self.batch_scheduler = batch_size_scheduler


    def action_selection(self, state):
        """
        Selects best action according to q-function for a given state
        param state: current state
        return: action with highest reward
        """
        with torch.no_grad():
            self.actor_network.eval()
            action = self.actor_network(state)
            self.actor_network.train()
            return action

    def soft_update(self, source, target):
        for target_w, source_w  in zip(target.parameters(), source.parameters()):
            target_w.data.copy_(
                (1.0 - self.tau) * target_w.data \
                + self.tau * source_w.data
            )

    def update_actor(self, loss):
        # update actor
        self.actor_optim.zero_grad()
        # print("actor_loss: ", actor_loss)
        loss.backward()
        self.actor_optim.step()

    def update_critic(self, loss):
        # update critic
        self.critic_optim.zero_grad()
        # print("critic loss: ", critic_loss)
        loss.backward(retain_graph=True)
        self.critic_optim.step()

    def trial(self):
        """
        Test the target actor in the environment
        return: average total reward
        """
        print("trial average total reward:")
        with torch.no_grad():
            episodes = 5
            average_reward = 0
            for episode in range(episodes):
                done = False
                obs = self.env.reset()
                total_reward = 0
                for t in range(self.trial_horizon):
                    obs = self.transformObservation(obs)
                    state = torch.tensor(obs, dtype=torch.float32).to(self.device)

                    action = self.actor_target(state).cpu().detach().numpy()
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                    #self.env.render()
                # calculate average reward with incremental average
                average_reward += total_reward/episodes
        print(average_reward)
        return average_reward

    def save_model(self, reward):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        torch.save(self.actor_target.state_dict(), os.path.join(self.dirname, "actortarget_{}".format(reward)))
        torch.save(self.critic_target.state_dict(), os.path.join(self.dirname, "critictarget_{}".format(reward)))
        #torch.save(self.critic_optim.state_dict(), os.path.join(self.dirname, "criticoptim"))
        #torch.save(self.actor_optim.state_dict(), os.path.join(self.dirname, "actoroptim"))

    def load_model(self):
       if not os.path.exists(os.path.join(self.dirname, "ddpg_critic_cartpolestab")):
           print("no model checkoutpoint found")
           return
       self.critic_target.load_state_dict(torch.load(os.path.join(self.dirname, "ddpg_critic_cartpolestab")))
       self.critic_target.load_state_dict(torch.load(os.path.join(self.dirname, "ddpg_critic_cartpolestab")))
       self.actor_network.load_state_dict(torch.load(os.path.join(self.dirname, "ddpg_actor_cartpolestab")))
       self.actor_network.load_state_dict(torch.load(os.path.join(self.dirname, "ddpg_actor_cartpolestab")))

    def update(self, e):
        sample_batch = self.replayBuffer.sample_batch(batch_size_schedulers[self.batch_scheduler](self.batch_size, e))
        s_batch, a_batch, r_batch, s_2_batch, done_batch = sample_batch

        # calculate policy/actor loss
        actor_loss = self.critic_network(s_batch, self.actor_network(s_batch))
        actor_loss = - actor_loss.mean() # it makes difference if you do gradient ascent or descent for your problem, you idiot

        # calculate value/critic loss
        next_action = self.actor_target(s_2_batch)
        critic_target_prediction = self.critic_target(s_2_batch, next_action)
        expected_critic = r_batch + self.gamma * (1. - done_batch) * critic_target_prediction

        critic_pred = self.critic_network(s_batch, a_batch)
        critic_loss = self.loss(critic_pred, expected_critic)

        return actor_loss, critic_loss

    def train(self):
        reward_record = 0
        print("Training started...")
        total_reward = 0
        episode = 0
        rew = []
        step = 0
        while step < self.total_steps:
            state = self.transformObservation(self.env.reset())
            done = False
            bs = batch_size_schedulers[self.batch_scheduler](self.batch_size, episode)
            statusprint = "{} /{} | {:.0f} /{:.0f} | {} /{} | alr,clr: {:.2E} {:.2E} | bs: {}"
            print(statusprint.format(step, self.total_steps, total_reward, reward_record, self.replayBuffer.count, self.replayBuffer.buffer_size, self.critic_lr_scheduler.get_lr()[0], self.actor_lr_scheduler.get_lr()[0], batch_size_schedulers[self.batch_scheduler](self.batch_size,episode)))
            total_reward = 0
            while not done:

                action = self.action_selection(torch.squeeze(torch.tensor(state, dtype=torch.float32, device=self.device)))

                action = self.noise_torch.sample((self.action_dim,)) *self.noise_decay**episode + action

                action = torch.clamp(action, min=self.env_low[0], max=self.env_high[0])

                action = action.to("cpu").detach().numpy()
                next_state, reward, done, _ = self.env.step(action)

                next_state = self.transformObservation(next_state)

                total_reward += reward
                self.replayBuffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.replayBuffer.count >= self.n_batches:

                    for _ in range(self.epochs):
                        actor_loss, critic_loss = self.update(episode)

                        self.update_actor(actor_loss)
                        self.update_critic(critic_loss)

                        self.soft_update(self.actor_network, self.actor_target)
                        self.soft_update(self.critic_network, self.critic_target)

                step += 1

            if self.replayBuffer.count >= self.n_batches:
#                print("critic loss: ", critic_loss)
#                print("actor_loss: ", actor_loss)
                self.critic_lr_scheduler.step()
                self.actor_lr_scheduler.step()
                episode += 1
                # if out actor is really good, test target actor. If the target actor is good too, save it.
                if reward_record < total_reward:
                    trial_average_reward = self.trial()
                    if trial_average_reward > reward_record:
                        print("New record")
                        reward_record = trial_average_reward
                        self.save_model(trial_average_reward)
                rew.append(total_reward)

        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.plot(rew)
        print(reward_record)
        plt.savefig(os.path.join(self.dirname, "rewardplot.png"))

        # test & save final model
        trial_average_reward = self.trial()
        self.save_model("{:.2f}_final".format(trial_average_reward))

        return reward_record
