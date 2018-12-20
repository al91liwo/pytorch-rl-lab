import copy
import torch
from torch import nn
import numpy as np
from pympler import summary, muppy
import gc

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer

gc.enable()

class DDPG:
    
    def __init__(self, env, buffer_size=1000000, batch_size=64,
                 epsilon=.99, tau=1e-2, episodes=100, warmup_samples=10000,
                 transform= lambda x : x, actor_lr=1e-4, critic_lr=1e-3):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        actor_param = [self.state_dim, 100, 200, 300, 300, self.action_dim]
        critic_param = [self.state_dim + self.action_dim, 100, 200, 300, 300, 1]
        self.actor_network = ActorNetwork(actor_param)
        self.critic_network = CriticNetwork(critic_param)
        self.actor_target = ActorNetwork(actor_param)
        self.critic_target = CriticNetwork(critic_param)
        
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        
        self.replayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_batches = warmup_samples

        self.epsilon = epsilon

        self.tau = tau
        self.episodes = episodes

        self.noise_warmup = torch.distributions.normal.Normal(0, self.env.action_space.high[0])
        self.noise_training = torch.distributions.normal.Normal(0, self.env.action_space.high[0]/5)
        self.transformObservation = transform

    def action_selection(self, state):
        """
        Selects best action according to q-function for a given state
        param state: current state
        return: action with highest reward
        """
        self.actor_target.eval()
        action = self.actor_target(state)
        self.actor_target.train()
        return action

    def softUpdate(self, source, target):
        for target_w, source_w  in zip(target.parameters(), source.parameters()):
            target_w.data.copy_(
                (1.0 - self.tau) * target_w.data \
                + self.tau * source_w.data
            )

    def update_actor(self, loss):
        # update actor
        self.actor_optim.zero_grad()
        loss.backward(retain_graph=False)
        self.actor_optim.step()

    def update_critic(self, loss):
        # update critic
        self.critic_optim.zero_grad()
        loss.backward(retain_graph=False)
        self.critic_optim.step()

    def train(self):
        print("Training started...")
        i = 0
        while i < self.episodes:
            total_reward = 0
            obs = self.env.reset()
            done = False
            step_i = 0

            print(i, self.episodes)
            while (not done):
                state = self.transformObservation(obs)

                action = self.action_selection(torch.squeeze(torch.tensor(state)))
                noise_function = self.noise_training if self.replayBuffer.count >= self.n_batches else self.noise_warmup
                action = noise_function.sample((self.action_dim,)) + action


                obs, reward, done, _ = self.env.step(np.array([action.item()]))
                next_state = self.transformObservation(obs)


                total_reward += reward


                # for continuity in replay buffer the next_state should should be a tensor
                self.replayBuffer.add(state, action, reward, next_state)
                if self.replayBuffer.count >= self.n_batches:
                        step_i = 1
                        sample_batch = self.replayBuffer.sample_batch(self.batch_size)
                        s_batch, a_batch, r_batch, s_2_batch = sample_batch

                        # calculate value/critic loss
                        next_action = self.actor_target(s_2_batch)
                        critic_target_prediction = self.critic_target(s_2_batch, next_action)
                        expected_critic = r_batch + self.epsilon * critic_target_prediction

                        critic_pred = self.critic_network(s_batch, a_batch)
                        critic_loss = self.loss(critic_pred, expected_critic)
                        self.update_critic(critic_loss)

                        # calculate policy/actor loss
                        actor_loss = self.actor_network(s_batch)
                        actor_loss = - self.critic_network(s_batch, actor_loss)
                        actor_loss = torch.mean(actor_loss)
                        self.update_actor(actor_loss)

                        self.softUpdate(self.critic_network, self.critic_target)
                        self.softUpdate(self.actor_network, self.actor_target)


            i += step_i

            if self.replayBuffer.count >= self.n_batches:
                print("critic loss: ", critic_loss)
                print("actor_loss: ", actor_loss)
                print(self.replayBuffer.count)
                gc.collect()
                # all_objects = muppy.get_objects()
                # sum = summary.summarize(all_objects)
                # summary.print_(sum)

                        
