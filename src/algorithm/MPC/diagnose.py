import torch.nn
import torch.distributions as distributions
import numpy as np
import time
from model import perfect_models
from matplotlib import pyplot as plt
from util import angle_from_sincos, angle_normalize

class ModelEval():
    """
    Uses reference dynamics and reward models to evaluate the performance of a model
    """
    def __init__(self, mpc, env, n_samples=20000):
        self.mpc = mpc
        self.env = env
        self.action_space_uniform = distributions.uniform.Uniform(torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

        self._generate_testdata(n_samples)

    def _generate_testdata(self, n_samples):
        samples = []
        done = False
        obs = self.env.reset()
        while not done and len(samples) < n_samples:
            action = self.action_space_uniform.sample()
            action = action.detach().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            obs = next_obs
        samples = np.array(samples)
        self.test_states_in = torch.from_numpy(np.vstack(samples[:,0])).float()
        self.test_actions_in = torch.from_numpy(np.vstack(samples[:,1])).float()
        self.test_states_out = torch.from_numpy(np.vstack(samples[:,3])).float()
        #self.test_rewards_out = torch.from_numpy(np.vstack(samples[:,2])).float()
        # self.test_inputs = torch.cat((states_in.float(), actions_in.float()), dim=1)
        # self.test_targets = torch.cat((states_out.float(), rewards_out.float()), dim=1)

    def plot_action_velocity(self, state):
        model = self.mpc.model
        # speed from state (cos,sin,velo) (0,-1,0) applying different action
        actions = torch.linspace(-2., 2., 4000).unsqueeze(1)
        states = torch.tensor(state).repeat(4000, 1)

        predictied_velo = model.propagate(states, actions)[:, 2]  # only get the velocity
        actual_velo = perfect_models['Pendulum-v0'].propagate(states, actions)[:, 2]

        plt.plot(actions.detach().numpy(), predictied_velo.detach().numpy(), label='prediction')
        plt.plot(actions.detach().numpy(), actual_velo.detach().numpy(), label='truth')
        plt.legend()

    def plot_action_state(self, state):
        model = self.mpc.model
        # speed from state (cos,sin,velo) (0,-1,0) applying different action
        actions = torch.linspace(-2., 2., 4000).unsqueeze(1)
        states = torch.tensor(state).repeat(4000, 1)

        out = model.propagate(states, actions)
        predicted_cos = out[:, 0]
        predicted_sin = out[:, 1]  # only get cos and sin
        out = perfect_models['Pendulum-v0'].propagate(states, actions)
        actual_cos = out[:, 0]
        actual_sin = out[:, 1]

        plt.plot(actions.detach().numpy(), predicted_cos.detach().numpy(), color='orange', label='cos prediction')
        plt.plot(actions.detach().numpy(), predicted_sin.detach().numpy(),color="red", label='sin prediction')
        plt.plot(actions.detach().numpy(), actual_cos.detach().numpy(), color="cyan", label='cos truth')
        plt.plot(actions.detach().numpy(), actual_sin.detach().numpy(), color="blue", label='sin truth')
        plt.legend()

    def eval_print(self):
        model = self.mpc.model
        self.model_states_out = model.propagate(self.test_states_in.float(), self.test_actions_in.float())
        err_dynamics = torch.mean((self.model_states_out - self.test_states_out) ** 2)
        print("Dynamics model mean squared error: ", err_dynamics)
        plt.figure(figsize=(20, 20))
        states = [[-1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., -1., 1.], [-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [-1., 0., -1.], [1., 0., -1.], [0., 1., -1.], [0., -1., -1.], [-1., 0., 4.], [1., 0., 4.], [0., 1., 4.], [0., -1., 4.]]
        for i, state in enumerate(states):
            ax = plt.subplot(4, 4, i+1)
            if i % 2 == 0:
                title = "VEL ang:{:.2f}, vel:{:.2f}".format(angle_from_sincos(torch.tensor(state[0]), torch.tensor(state[1])), state[2])
                self.plot_action_velocity(state)
            else:
                title = "STA ang:{:.2f}, vel:{:.2f}".format(angle_from_sincos(torch.tensor(state[0]), torch.tensor(state[1])), state[2])
                self.plot_action_state(state)
            ax.set_title(title)
        plt.show()

class TimeDiagnoser():
    """
    TimeDiagnoser can be used to analyze code performance by making time logging easy.
    """

    def __init__(self, *labels):
        self.labels = labels
        self.times = {}
        self.start_times = {}
        for label in labels:
            self.times[label] = []
            self.start_times[label] = 0

    def log_time(self, label, time):
        self.times[label].append(time)

    def start_log(self, label):
        self.start_times[label] = time.time()

    def end_log(self, label):
        self.times[label].append(time.time() - self.start_times[label])

    def print_times(self):
        for l, t in self.times.items():
            print(l, ": ", str(sum(t) / len(t)))

    def print_label(self, label):
        print(label, ": ", sum(self.times[label]) / len(self.times[label]))

    def reset_times(self):
        for label in self.labels:
            self.times[label] = []
            self.start_times[label] = 0