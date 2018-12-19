import copy
import torch
from torch import nn

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer



class DDPG:
    
    def __init__(self, env, buffer_size=1000000, batch_size=64,
                 epsilon=.99, tau=1e-2, episodes=100, warmup_samples=10000,
                 transform= lambda x : x, actor_lr=1e-3, critic_lr=1e-3):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.actor_network = ActorNetwork([self.state_dim, 100, 150, 200, 250, 300, self.action_dim])
        self.critic_network = CriticNetwork([self.state_dim + self.action_dim, 100, 150, 200, 250, 300, 1])
        self.actor_target = copy.deepcopy(self.actor_network)
        self.critic_target = copy.deepcopy(self.critic_network)
        
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        
        self.replayBuffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_batches = warmup_samples

        self.epsilon = epsilon

        self.tau = tau
        self.episodes = episodes
        print(self.env.action_space.high)
        self.noise_torch= torch.distributions.normal.Normal(0, self.env.action_space.high[0])
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
        # print("actor_loss: ", actor_loss)
        loss.backward()
        self.actor_optim.step()

    def update_critic(self, loss):
        # update critic
        self.critic_optim.zero_grad()
        # print("critic loss: ", critic_loss)
        loss.backward(retain_graph=True)
        self.critic_optim.step()

    def train(self):
        print("Training started...")

        for step in range(self.episodes):
            total_reward = 0
            state = torch.from_numpy(self.transformObservation(self.env.reset())).type(torch.FloatTensor)
            done = False
            print(step, self.episodes)
            while (not done):
                action = self.action_selection(state)
                action = self.noise_torch.sample((self.action_dim,)) + action

                next_state, reward, done, _ = self.env.step([action.item()])
                next_state = self.transformObservation(next_state)
                next_state = torch.squeeze(torch.from_numpy(next_state).type(torch.FloatTensor))
                total_reward += reward


                # for continuity in replay buffer the next_state should should be a tensor
                self.replayBuffer.add(state, action, reward, next_state)
                state = next_state
                if self.replayBuffer.count >= self.n_batches:
                        sample_batch = self.replayBuffer.sample_batch(self.batch_size)
                        s_batch, a_batch, r_batch, s_2_batch = sample_batch

                        # calculate policy/actor loss
                        actor_loss = self.critic_network(s_batch, self.actor_network(s_batch))
                        actor_loss = - actor_loss.mean()

                        # calculate value/critic loss
                        next_action = self.actor_target(s_2_batch)
                        critic_target_prediction = self.critic_target(s_2_batch, next_action)
                        expected_critic = r_batch + self.epsilon * critic_target_prediction

                        critic_pred = self.critic_network(s_batch, a_batch)
                        critic_loss = self.loss(critic_pred, expected_critic)

                        self.update_actor(actor_loss)
                        self.update_critic(critic_loss)

                        self.softUpdate(self.actor_network, self.actor_target)
                        self.softUpdate(self.critic_network, self.critic_target)

            if self.replayBuffer.count >= self.n_batches:
                print("critic loss: ", critic_loss)
                print("actor_loss: ", actor_loss)
            

                        
