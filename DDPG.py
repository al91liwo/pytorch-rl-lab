import copy
import torch
from torch import nn
import matplotlib.pyplot as plt

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer
import os
import datetime


class DDPG:
    
    def __init__(self, env, buffer_size=10000, batch_size=64,
                 gamma=.99, tau=1e-2, episodes=50, warmup_samples=1000, noise_decay=0.9,
                 transform=lambda x: x, actor_lr=1e-3, critic_lr=1e-3, trial_horizon=1000,
                 actor_hidden_layers=[10, 10, 10], critic_hidden_layers=[10, 10, 10]):
        self.env = env
        self.env_low = self.env.action_space.low
        self.env_high = self.env.action_space.high
        print(self.env_low, self.env_high)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.started = datetime.datetime.now()
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.warmup_samples = warmup_samples

        self.actor_network = ActorNetwork([self.state_dim, *actor_hidden_layers, self.action_dim], torch.tensor(self.env_low), torch.tensor(self.env_high))
        self.critic_network = CriticNetwork([self.state_dim + self.action_dim, *critic_hidden_layers, 1])
        self.actor_target = copy.deepcopy(self.actor_network)
        self.critic_target = copy.deepcopy(self.critic_network)
        
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        self.noise_decay = noise_decay
        self.trial_horizon = trial_horizon
        
        self.replayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_batches = warmup_samples

        self.gamma = gamma

        self.tau = tau
        self.episodes = episodes
        self.noise_torch = torch.distributions.normal.Normal(0, self.env_high[0])
        self.transformObservation = transform

    def action_selection(self, state):
        """
        Selects best action according to q-function for a given state
        param state: current state
        return: action with highest reward
        """ 
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
        episodes = 2
        for step in range(episodes):
            done = False
            obs = self.env.reset()[0]
            total_reward = 0
            print(step)
            while not done:
                obs = self.transformObservation(obs)
                state = torch.tensor(obs, dtype=torch.float32)

                action = self.actor_target(state).detach().numpy()
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.env.render()
            print(total_reward)

    def save_model(self, env_name, actor_path=None, critic_path=None):
        suffix = self.started
        if not os.path.exists('models/{}'.format(suffix)):
            os.makedirs('models/{}'.format(suffix))
        if actor_path is None:
            actor_path = "models/{}/ddpg_actor_{}".format(suffix, env_name)
        if critic_path is None:
            critic_path = "models/{}/ddpg_critic_{}".format(suffix, env_name)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor_network.state_dict(), actor_path)
        torch.save(self.critic_network.state_dict(), critic_path)

    def update(self):
        sample_batch = self.replayBuffer.sample_batch(self.batch_size)
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
        print("Training started...")
        total_reward = 0
        episode = 0
        rew = []
        while episode < self.episodes:

            state = self.transformObservation(self.env.reset()[0])
            done = False
            print(episode, "/", self.episodes, "|", total_reward, "|", self.replayBuffer.count, "/", self.replayBuffer.buffer_size)
            total_reward = 0
            step = 0
            while not done:

                action = self.action_selection(torch.squeeze(torch.tensor(state, dtype=torch.float32)))

                action = self.noise_torch.sample((self.action_dim,)) *self.noise_decay**episode + action

                action = torch.clamp(action, min=self.env_low[0].item(), max=self.env_high[0].item())

                action = action.detach().numpy()
                next_state, reward, done, _ = self.env.step(action)

                next_state = self.transformObservation(next_state)

                total_reward += reward
                action = [action.item()]
                self.replayBuffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.replayBuffer.count >= self.n_batches:

                    actor_loss, critic_loss = self.update()

                    self.update_actor(actor_loss)
                    self.update_critic(critic_loss)

                    self.soft_update(self.actor_network, self.actor_target)
                    self.soft_update(self.critic_network, self.critic_target)

                step += 1

            if self.replayBuffer.count >= self.n_batches:
                print("critic loss: ", critic_loss)
                print("actor_loss: ", actor_loss)
                episode += 1
                if total_reward > 1000:
                    self.trial()
                rew.append(total_reward)

        plt.plot(range(self.episodes), rew)
        if not os.path.exists('models/{}'.format(self.started)):
            os.makedirs('models/{}'.format(self.started))
        plt.savefig('models/{}/buffer_size={}_batch_size={}_gamma={}_tau={}_episodes={}_warmup_samples={}_\
        noise_decay={}_actor_lr={}_critic_lr={}_actor_hidden_layers={}_critic_hidden_layers={}.jpg'.format(self.started, self.buffer_size, self.batch_size, self.gamma, self.tau, self.episodes, self.warmup_samples, self.noise_decay, self.actor_lr, self.critic_lr, self.actor_hidden_layers, self.critic_hidden_layers))