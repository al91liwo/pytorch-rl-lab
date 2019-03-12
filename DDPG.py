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


class DDPG:
    
    def __init__(self, env, action_space_limits, dirname="out", buffer_size=10000, batch_size=64, is_quanser_env=True,
                 gamma=.99, tau=1e-2, steps=100000, warmup_samples=1000, noise_decay=0.9,
                 transform=lambda x: x, actor_lr=1e-3, critic_lr=1e-3, lr_decay=1.0, lr_min=1.e-7, trial_horizon=5000,
                 actor_hidden_layers=[10, 10, 10], critic_hidden_layers=[10, 10, 10], device="cpu"):
        """
        DDPG algorithm implementation as in TODO: link
        param env: the gym environment to deal with
        param dirname: non-existing or existing directory in which the calculated immediate models will be saved in
        param action_space_limits: sets a limit on action space
        param buffer_size: the size of the replay buffer 
        param batch_size: size of batches to learn with while training, extracted from replay buffer
        param is_quaner_env: True if given env is from quaner_robots, else false
        param gamma: interest rate of expected return
        param tau: update factor from source to target network
        param steps: number of steps that will be performed during training time
        param warmup_samples: number of random samples placed into replay buffer before training actor and critic network
        param noise_decay: gaussian noise on actions will be reduced multiplicative in every episode by this factor
        param transform: function to transform observation space of given environment
        param actor_lr: learning rate of adam optimizer for actor network
        param critic_lr: learning rate of adam optimizer for critic network
        param lr_decay: learning rate decay of adam optimizers
        param lr_min: lower bound of learning rate of adam_optimizers
        param trial_horizon: maximum steps to take per episode
        param actor_hidden_layers: hidden layers of actor network as a numeric list
        param critic_hidden_layers: hidden layers of critic network as a numeric list
        param device: on which device to train your torch nn.Models on either cpu or gpu
        """
        self.device = device
        #algorithm timestamp
        self.started = datetime.datetime.now()
        
        self.env = env
        self.is_quanser_env = is_quanser_env
        self.dirname = dirname
        self.env_low = torch.tensor(action_space_limits[0], device=self.device)
        self.env_high = torch.tensor(action_space_limits[1], device=self.device)
        self.warmup_samples = warmup_samples
        self.total_steps = steps
        self.transformObservation = transform

        #replay buffer parameters + initialization
        self.buffer_size = buffer_size
        self.replayBuffer = ReplayBuffer(self.buffer_size, self.device)
        self.batch_size = batch_size
        self.n_batches = warmup_samples

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        #optimizer parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
  
        #actor and critic parameters + initialization      
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.actor_network = ActorNetwork([self.state_dim, *self.actor_hidden_layers, self.action_dim], torch.tensor(self.env_low, device=self.device), torch.tensor(self.env_high, device=self.device)).to(self.device)
        self.critic_network = CriticNetwork([self.state_dim + self.action_dim, *self.critic_hidden_layers, 1]).to(self.device)
        self.actor_target = copy.deepcopy(self.actor_network).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_network).to(self.device)
        
        #optimizer initialization
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optim, lr_decay)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optim, lr_decay)
        
        #training parameters
        self.loss = nn.MSELoss()
        self.noise_decay = torch.tensor(noise_decay, device=self.device)
        self.trial_horizon = trial_horizon
        self.gamma = torch.tensor(gamma, device=self.device)
        self.tau = torch.tensor(tau, device=self.device)
        #gaussian noise on actions used
        self.noise_torch = torch.distributions.normal.Normal(0, self.env_high[0])



    def action_selection(self, state):
        """
        Selects best action according to q-function for a given state
        param state: current state
        return: action with highest q-value
        """
        with torch.no_grad():
            self.actor_network.eval()
            action = self.actor_network(state)
            self.actor_network.train()
            return action

    def soft_update(self, source, target):
        """
        Updates the weights of given target network (nn.Module) by the weights of given source network (nn.Module) 
        param source: nn.Module which weights will be taken from
        param target: nn.Module which weights will be updated to
        """
        for target_w, source_w  in zip(target.parameters(), source.parameters()):
            target_w.data.copy_(
                (1.0 - self.tau) * target_w.data \
                + self.tau * source_w.data
            )

    def update_actor(self, loss):
        """
        Updates actor network by given calculated loss
        """
        # update actor
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def update_critic(self, loss):
        """
        Updates critic network by given calculated loss
        """
        # update critic
        self.critic_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.critic_optim.step()
   
    def forwardActorNetwork(self, network, state):
        """
        Forwards state through either target or training ActorNetwork
        param network: either target or training ActorNetwork
        param state: state to forward through network
        return: action for environment step 
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        action = network(state).squeeze()
        #dimensionality check of actions
        action = action.unsqueeze(0).cpu().detach().numpy() if action.dim() == 0 else action
        if self.is_quanser_env:
            action = np.array(action)
        return action

    def trial(self):
        """
        Test the target actor in the environment
        return: average total reward
        """
        print("trial average total reward:")
        self.actor_target.eval()
        with torch.no_grad():
            episodes = 5
            average_reward = 0
            for episode in range(episodes):
                done = False

                obs = self.env.reset()
                total_reward = 0
                for t in range(self.trial_horizon):
                    state = self.transformObservation(obs)
                    action = self.forwardActorNetwork(self.actor_target, state)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    if done:
                        break
                # calculate average reward with incremental average
                average_reward += total_reward/episodes
        print(average_reward)
        self.actor_target.train()
        return average_reward

    def save_model(self, reward):
        """
        Saves the immediate actor and critic target network in given self.dirame directory
        param reward: will be displayed as filename
        """
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        torch.save(self.actor_target.state_dict(), os.path.join(self.dirname, "actortarget_{}".format(reward)))
        torch.save(self.critic_target.state_dict(), os.path.join(self.dirname, "critictarget_{}".format(reward)))
        #torch.save(self.critic_optim.state_dict(), os.path.join(self.dirname, "criticoptim"))
        #torch.save(self.actor_optim.state_dict(), os.path.join(self.dirname, "actoroptim"))

    def update(self):
        """
        Calculating loss w.r.t. DDPG paper TODO: link paper 
        return: actor and critic loss        
        """
        sample_batch = self.replayBuffer.sample_batch(self.batch_size)
        s_batch, a_batch, r_batch, s_2_batch, done_batch = sample_batch


        # calculate policy/actor loss
        actor_loss = self.critic_network(s_batch, self.actor_network(s_batch))
        actor_loss = - actor_loss.mean() 

        # calculate value/critic loss
        next_action = self.actor_target(s_2_batch)
        critic_target_prediction = self.critic_target(s_2_batch, next_action)
        expected_critic = r_batch + self.gamma * (1. - done_batch) * critic_target_prediction

        critic_pred = self.critic_network(s_batch, a_batch)
        critic_loss = self.loss(critic_pred, expected_critic)

        return actor_loss, critic_loss

    def info_print(self, step, total_reward, reward_record):
        """
        Status print of this training session per episode
        """
        statusprint = "{} /{} | {:.0f} /{:.0f} | {} /{} | alr,clr: {:.2E} {:.2E}"
        print(statusprint.format(step, self.total_steps, total_reward, reward_record, self.replayBuffer.count, self.replayBuffer.buffer_size, self.actor_lr_scheduler.get_lr()[0], self.critic_lr_scheduler.get_lr()[0]))
  

    def train(self):
        """
        A training session w.r.t. to training parameters
        return: the best reward achieved during this training session
        """
        print("Training started...")
        reward_record = 0
        total_reward = 0
        episode = 0
        rew = []
        step = 0
        while step < self.total_steps:
            state = self.transformObservation(self.env.reset())
            done = False
    
            self.info_print(step, total_reward, reward_record)
            
            total_reward = 0
            i = 0
            while not done:
              
                action = self.action_selection(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze()

                action = self.noise_torch.sample((self.action_dim,)) *self.noise_decay**episode + action

                action = torch.clamp(action, min=self.env_low[0], max=self.env_high[0])

                action = action.to("cpu").detach().numpy()
                next_state, reward, done, _ = self.env.step(action)
                done = done or i >= self.trial_horizon
                next_state = self.transformObservation(next_state)
    
                total_reward += reward
            
                self.replayBuffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.replayBuffer.count >= self.n_batches:

                    actor_loss, critic_loss = self.update()

                    self.update_actor(actor_loss)
                    self.update_critic(critic_loss)

                    self.soft_update(self.actor_network, self.actor_target)
                    self.soft_update(self.critic_network, self.critic_target)

                step += 1
                i = i + 1
            if self.replayBuffer.count >= self.n_batches:
                if self.critic_lr_scheduler.get_lr()[0] > self.lr_min:
                    self.critic_lr_scheduler.step()
                if self.actor_lr_scheduler.get_lr()[0] > self.lr_min:
                    self.actor_lr_scheduler.step()
                episode += 1
                # if out actor is really good, test target actor. If the target actor is good too, save it.
                if reward_record < total_reward and total_reward > 50:
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
        plt.clr()
        
        # write plot data to a file
        with open(os.path.join(self.dirname, "rewarddata"), "w+") as f:
            f.write(','.join([str(reward) for reward in rew]))

        # test & save final model
        trial_average_reward = self.trial()
        self.save_model("{:.2f}_final".format(trial_average_reward))

        return reward_record
