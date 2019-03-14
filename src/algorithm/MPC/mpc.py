import numpy as np
import torch
import torch.distributions as distributions

from src.algorithm.MPC.diagnose import ModelEval
from src.algorithm.MPC.control import TrajectoryController
from src.algorithm.MPC.control import control_time_diagnoser


class MPC:
    """
    MPC implements Model Predictive Control algorithm for reinforcement learning.
    During the first trials it gathers data from the environment by using a random control policy.
    After enough data has been gathered the model of the environment is trained.
    Successive trials will use the CEM algorithm to find an optimal trajectory and execute the
    first action of the best trajectory. After each trial, the model is trained using the new data.
    """

    def __init__(self, env, reward, model, trainer, predict_horizon=20, warmup_trials=1, learning_trials=20,
                 trial_horizon=1000, cem_samples=400, nelite=0, render=0, device="cpu", max_memory=1000000):
        """
        Creates a Model Predictive Controller
        :param env: an OpenAI Gym environment
        "param reward: the reward function
        :param model: a trainable model for the environment
        :param trainer: an algorithm to train the model
        :param predict_horizon: number of steps to look ahead when optimizing the trajectory
        :param warmup_trials: the number of trials with random controller before starting to use trajectory planning
        :param learning_trials: the number of trials to keep explorating. Afterwards, the controller will exploit.
        :param trial_horizon: the maximum amount of steps per trial
        :param cem_samples: the number of random samples to evaluate with cem
        :param render: modulus for the trials to render. 0 means no render
        """
        self.env = env
        self.reward = reward
        self.model = model
        self.trainer = trainer
        self.predict_horizon=predict_horizon
        self.warmup_trials = warmup_trials
        self.learning_trials=learning_trials
        self.trial_horizon = trial_horizon
        self.render = render
        self.device = device
        self.memory = None
        self.action_space_dim = env.action_space.shape[0]
        self.action_space_min = torch.from_numpy(self.env.action_space.low).to(device)
        self.action_space_max = torch.from_numpy(self.env.action_space.high).to(device)
        self.action_space_uniform = distributions.uniform.Uniform(self.action_space_min, self.action_space_max)
        self.state = env.reset()
        self.diagnose_model = ModelEval(self, self.env)
        self.max_memory = max_memory
        self.trajectory_controller = TrajectoryController(self.model, self.reward, self.action_space_dim,
                                                          self.action_space_min,
                                                          self.action_space_max, self.predict_horizon,
                                                          self.predict_horizon, self._expected_reward, self.device,
                                                          cem_samples=cem_samples, nelite=nelite)

    def _expected_reward(self, action_trajectory):
        """
        Calculates the expected rewards over for a given trajectory of actions, starting from self.state and propagating
        with self.model.
        :param action_trajectory: 2D tensor of actions(one trajectory) or 3D tensor of a batch of trajectories
        :return: the expected reward or a tensor of expected rewards
        """
        with torch.no_grad():
            traj_dim = action_trajectory.dim()
            reward = 0
            state = torch.tensor(self.state, requires_grad=False).to(self.device)

            # check whether the input is a batch of trajectories
            # if this is the case, we need to create a vector of state particles
            if traj_dim > 2:
                # if this is the case, we need to create a vector of state particles
                state = state.repeat(len(action_trajectory), 1)
                # makes iterating over it easy, as we have the n-th action of each trajectory in the same row
                action_trajectory = torch.transpose(action_trajectory, 0, 1)
            # action is either a single action or a vector of actions, depending on whether action_trajectory is a batch
            for action in action_trajectory:  # unsqueeze needed so action is not 0-dim
                state = self.model.propagate(state.float(), action)
                next_reward = self.reward(state, action)
                reward += next_reward
            return reward

    def _memory_push(self, samples):
        if self.memory is None:
            self.memory = np.array(samples)
        else:
            overflow = len(samples) + len(self.memory) - self.max_memory
            if overflow > 0:
                self.memory = self.memory[overflow:] # cut away from memory to make space for new samples
            self.memory = np.vstack((self.memory, samples))

    def _trial(self, controller, horizon=0, render=False):
        """
        Runs a trial on the environment. Renders the environment if self.render is True.
        :param controller: provides the next action to take
        :param horizon: the maximum steps to take. 0 means infinite steps.
        :param render: whether to render the current trial
        :return: cummulative reward
        """
        if not self.trajectory_controller is None:
            self.trajectory_controller.reset()
        # horizon=0 means infinite horizon
        obs = self.env.reset()
        self.state = obs
        samples = []
        cum_reward = 0
        t = 0
        while True:
            if render:
                self.env.render()
            action = controller(torch.tensor(obs, device=self.device))
            # print(action)
            action = action.detach().numpy()
            # print(action)
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            cum_reward += reward
            t += 1
            if done or horizon > 0 and t == horizon:
                break
            obs = next_obs
            self.state = next_obs

        self._memory_push(np.array(samples))

        return cum_reward

    def _train_model(self, epochs=0):
        """
        Trains the model with data from self.memory and self.trainer.
        Transforms the numpy arrays from the environment to torch tensors.
        """
        states_in = torch.from_numpy(np.vstack(self.memory[:, 0])).to(self.device)
        actions_in = torch.from_numpy(np.vstack(self.memory[:, 1])).to(self.device)
        states_out = torch.from_numpy(np.vstack(self.memory[:, 3])).to(self.device)
        # we do not need rewards as we changed to reward less model
        # rewards_out = torch.from_numpy(np.vstack(self.memory[:, 2])).to(self.device)
        inputs = torch.cat((states_in.float(), actions_in.float()), dim=1)
        # targets = torch.cat((states_out.float(), rewards_out.float()), dim=1)

        if epochs == 0:
            self.trainer.train(inputs, states_out)
        else:
            self.trainer.train(inputs, states_out, epochs)

    def _random_controller(self, obs, n=1):
        """
        Controller that generates random actios that respect the action space.
        :param obs: not used, exists to match the implicit controller interface
        :param n: number of actions to sample.
        :return: Sampled action. If n is 1, the action will no be packed in a list.
        """
        if n <= 0:
            raise Exception("number of samples as to be greater than 0")
        if n == 1:
            return self.action_space_uniform.sample().to(self.device)
        else:
            return self.action_space_uniform.sample((n,)).to(self.device)

    def _trajectory_controller(self, obs):
        """
        _trajectory_controller uses the model to optimize upon the next self.predict_horizon actions.
        The first action of the optimal trajectory is returned.
        :param obs: current observation of the environment.
        :return: the next action ot take
        """
        self.trajectory_controller.cost_func = self._expected_reward
        return self.trajectory_controller.next_action(obs)

    def train(self, diagnose=False):
        """
        Starts the reinforcement learning algorithm on the environment.
        """
        for k in range(self.warmup_trials):
            print("Warmup trial #", k)
            self._trial(self._random_controller)

        print("Initial training after warmup.")
        # initial training uses a higher amount of epochs
        self._train_model(5)

        for k in range(self.learning_trials):
            print("Learning trial #", k)
            reward = self._trial(self._trajectory_controller, self.trial_horizon, self.render > 0 and k % self.render == 0)
            print("Reward: ", reward)
            print(len(self.memory), " samples in buffer")
            print("Training after trial #", k)
            if diagnose:
                self.diagnose_model.eval_print()
                control_time_diagnoser.print_times()
                control_time_diagnoser.reset_times()
            self._train_model()
            # self.trajectory_controller.set_trajectory_len(self.trajectory_controller.trajectory_len + 1)
