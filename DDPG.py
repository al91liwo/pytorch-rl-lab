import copy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import quanser_robots
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer


class DDPG:
    
    def __init__(self, env, buffer_size=1000, batch_size=64,
            discount=0.99, epsilon=1., decrease=1e-4, tau=1e-3,
            episodes=500, n_batches=64,
            noise_func=np.random.rand, 
            transform= lambda x : x, actor_lr=1e-4, critic_lr=1e-4):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor_network = ActorNetwork(self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.state_dim, self.action_dim)
        self.actor_target = copy.deepcopy(self.actor_network)
        self.critic_target = copy.deepcopy(self.critic_network)
        
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        
        self.replayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_batches = n_batches
        
        self.discount = discount
        self.epsilon = epsilon
        self.decrease = decrease
        self.tau = tau
        self.episodes = episodes
        
        self.noise_torch=noise_func
        self.transformObservation = transform

    def action_selection(self, state):
        """
        Selects best action according to q-function for a given state
        param state: current state
        return: action with highest reward
        """ 
        self.actor_network.eval()
        action = self.actor_network(state)
        exploration_noise = self.epsilon * self.noise_torch(1)
        self.actor_network.train()
        return action + torch.from_numpy(exploration_noise).type(torch.FloatTensor)


    def softUpdate(self, source, target):
        for target_w, source_w  in zip(target.parameters(), source.parameters()):
            target_w.data.copy_(
                (1.0 - self.tau) * target_w.data \
                + self.tau * source_w.data
            )



    def train(self):
        print("Training started...")        

        for step in range(self.episodes):
            obs = self.env.reset()
            total_reward = 0
            done = False
            print(step, self.episodes)
            while (not done):
                obs = self.transformObservation(obs)
                
                state = np.reshape(np.array(obs), (1, self.state_dim))
                state = torch.from_numpy(state).type(torch.FloatTensor)
                        
                        
                
                action = self.action_selection(state)
                

                action = action.data.item()

                obs, reward, done, _ = self.env.step([action])
                obs = self.transformObservation(obs)
                 
                total_reward += reward

                # for continuity in replay buffer obs should be a tensor
                self.replayBuffer.add(state, action, reward, 
                        torch.from_numpy(
                            np.reshape(np.array(obs), (1, self.state_dim))
                            ).type(torch.FloatTensor)
                        )
                if self.replayBuffer.count >= self.n_batches:
                        sample_batch = self.replayBuffer.sample_batch(self.batch_size)
                        s_batch, a_batch, r_batch, s_2_batch = sample_batch

                        r_batch = torch.from_numpy(r_batch).type(torch.FloatTensor)
                        a_batch = torch.from_numpy(a_batch).type(torch.FloatTensor)
                        s_batch = torch.cat(s_batch, dim=0)
                        s_2_batch = torch.cat(s_2_batch, dim=0)
                   
                        # calculates prediction of critic_network
                        critic_Pred = self.critic_network(s_batch, a_batch)
                        actor_Tar_Pred = self.actor_target(s_2_batch)
                        
                        # learning parameter
                        y_i = r_batch + self.discount * self.critic_target(s_2_batch, actor_Tar_Pred)

                        # update critic    
                        self.critic_optim.zero_grad()
                        critic_loss = self.loss(y_i, critic_Pred)
                        #print("critic loss: ", critic_loss)
                        critic_loss.backward()
                        self.critic_optim.step()

                        # calculates prediction of actor_network
                        actor_Pred = self.actor_network(s_batch)

                        # update actor
                        self.actor_optim.zero_grad()
                        actor_loss = -self.critic_network(s_batch, actor_Pred)
                        actor_loss = actor_loss.mean()
                                             
                        #print("actor_loss: ", actor_loss)
                        actor_loss.backward()   
                        self.actor_optim.step()
                        
                        self.softUpdate(self.actor_network, self.actor_target)
                        self.softUpdate(self.critic_network, self.critic_target)
                        
                        #TODO: new epsilon decrease pls :(
                        self.epsilon -= self.decrease
            print("critic loss: ", critic_loss)
            print("actor_loss: ", actor_loss)


                        
