import torch
import torch.distributions as distributions
from src.algorithm.MPC.diagnose import TimeDiagnoser

control_time_diagnoser = TimeDiagnoser("average_cem_time")


def cem_optimize(init_mean, cost_func, init_variance=1., samples=400, precision=1.0e-3, steps=5, nelite=40, alpha=0.1,
                 contraint_mean=None, constraint_variance=(-999999, 999999), device="cpu"):
    """
    cem_optimize minimizes cost_function by iteratively sampling values around the current mean with a set variance.
    Of the sampled values the mean of the nelite number of samples with the lowest cost is the new mean for the next iteration.
    Convergence is met when either the change of the mean during the last iteration is less then precision.
    Or when the maximum number of steps was taken.
    :param init_mean: initial mean to sample new values around
    :param cost_func: varience used for sampling
    :param init_variance: initial variance
    :param samples: number of samples to take around the mean. Ratio of samples to elites is important.
    :param precision: if the change of mean after an iteration is less than precision convergence is met
    :param steps: number of steps
    :param nelite: number of best samples whose mean will be the mean for the next iteration
    :param alpha: softupdate, weight for old mean and variance
    :param contraint_mean: tuple with minimum and maximum mean
    :param constraint_variance: tuple with minumum and maximum variance
    :param device: either gpu or cpu (torch tensor configuration)
    :return:
    """
    control_time_diagnoser.start_log("average_cem_time")
    mean = init_mean
    covariance_matrizies = torch.stack([torch.diagflat(torch.tensor([init_variance], device=device)) for _ in range(len(mean))])
    # print(mean.type(), variance.type())
    step = 0
    diff = 9999999
    while diff > precision and step <= steps:
        # we create a distribution with action dimensionality and a batch size corresponding the trajectory length
        # dist.batch_shape == trajectory_len, dist.event_shape == action_space_dim
        dist = distributions.MultivariateNormal(mean, covariance_matrix=covariance_matrizies)
        candidates = dist.sample_n(samples).to(device)
        costs = cost_func(candidates)
        # we sort descending because we want a maximum reward
        sorted_idx = torch.argsort(costs, dim=0, descending=True)
        candidates = candidates[sorted_idx]
        elite = candidates[:nelite]
        new_mean = torch.mean(elite, dim=0)
        new_covariance_matrizies = torch.stack([torch.diagflat(v) for v in torch.var(elite, dim=0)])
        # calculate diff for break condition on precision
        diff = torch.mean(torch.abs(mean - new_mean))
        # softupdate mean and variance with alpha
        mean = (1 - alpha) * new_mean + alpha * mean
        covariance_matrizies = (1 - alpha) * new_covariance_matrizies + alpha * covariance_matrizies
        # print(mean, variance)
        if not contraint_mean is None:
            mean = clip(mean, contraint_mean[0], contraint_mean[1])
        step += 1
    control_time_diagnoser.end_log("average_cem_time")
    return mean


class FIFOBuffer:

    def __init__(self, length):
        self.length = length
        self.buffer = []

    def push(self, elem):
        self.buffer.insert(0, elem)
        if len(self.buffer) > self.length:
            del self.buffer[-1]

    def get(self):
        return self.buffer

    def clear(self):
        self.buffer.clear()


def clip(x, min, max):
    return torch.max(torch.min(x, max), min)


class TrajectoryController:

    def __init__(self, model, reward, action_dim, action_min, action_max, trajectory_len, history_len, cost_function, device, cem_samples=400, nelite=0):
        """
        A TrajectoryController finds the next best action by evaluating a trajectory of future actions.
        Future actions are evaluated using a model.
        It also takes history_len past actions into consideration.
        :param model: the system dynamics model to user for controlling
        :param reward: the reward function
        :param action_dim: the dimension of the action space
        :param trajectory_len: the number of future actions to look at
        :param history_len: the number of past actions to store
        :param cost_function: function (currentState, trajectory) -> expected reward that gives the cost for a trajectory
        """
        self.model = model
        self.reward = reward
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.trajectory_len = trajectory_len
        self.trajectory_shape = (self.trajectory_len, self.action_dim)
        self.history_len = history_len
        self.history = FIFOBuffer(self.history_len)
        self.cost_func = cost_function
        self.trajectory = None
        self.device = device
        self.cem_samples = cem_samples
        self.nelite = self.cem_samples // 10 if nelite == 0 else nelite

    def reset(self):
        self.trajectory = None

    def set_trajectory_len(self, new_traj_len):
        self.trajectory_len = new_traj_len
        self.trajectory_shape = (self.trajectory_len, self.action_dim)

    def next_action(self, obs):
        # history = self.history.get()
        # missing = self.history_len - len(history)
        # if missing == self.history_len:
        #     past_trajectory = torch.zeros(missing, self.action_dim)
        # elif missing > 0:
        #     past_trajectory = torch.cat((torch.stack(self.history.get()), torch.zeros(missing, self.action_dim)))
        # else:
        #     past_trajectory = torch.stack(self.history.get())
        if self.trajectory is None:
            # initialize trajectory
            self.trajectory = torch.zeros(self.trajectory_shape, device=self.device)
        else:
            self.trajectory = torch.cat((self.trajectory[1:], torch.zeros((1, self.action_dim), device=self.device)), dim=0)
        # find a trajectory that optimizes the cummulative reward

        self.trajectory = cem_optimize(self.trajectory, self.cost_func, init_variance=torch.max(torch.sqrt(self.action_max - self.action_min)).item(), contraint_mean=[self.action_min, self.action_max], device=self.device, steps=5, samples=self.cem_samples, nelite=self.nelite)
        # print(self.trajectory)
        best_action = self.trajectory[0]

        # self.history.push(best_action)
        return best_action
