import torch
import torch.nn as nn
import numpy as np
from src.utility.util import angle_from_sincos
import matplotlib.pyplot as plt


class Swish(torch.nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Swish}(x) = x * text{Sigmoid}(x) = x * \frac{1}{1 + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = Swish()
        >>> inp = torch.randn(2)
        >>> output = m(inp)
    """
    def forward(self, inp):
        return inp * torch.sigmoid(inp)


class NegLogLikelihood(nn.Module):
    @staticmethod
    def forward(output, target):
        """
        Computes the negative log likelihood prediction error over output and target(or a batch of each).
        The
        :param output: outputs of the entity to optimize
        :param target: the targets of the optimization
        :return: the mean log likelihood prediction error over the inputs
        """
        softplus = torch.nn.Softplus()
        mean, diagonalcovariance = torch.chunk(output, 2, dim=-1)
        # diagonalcovariance = torch.abs(diagonalcovariance)
        mean_err = mean - target

        loss = torch.mean(
            (mean_err * mean_err)/diagonalcovariance, dim=-1) + torch.mean(torch.log(diagonalcovariance), dim=-1)
        return loss.mean()


class NN(nn.Module):
    """
    A plain neural network.
    """
    def __init__(self, layers, activations, batch_norm=True):
        """
        Creates a neural network.
        :param layers: layers of the network, including input and output layers
        :param activations: per layer activation functions. If less then layers, default Swish will be used
        :param batch_norm: whether to batch normalize between layers
        """
        super(NN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        # standard activation function for each hidden layer
        self.activations = [Swish()]*(len(layers)-2)
        # override with array of passed in activation functions
        for i in range(len(activations)):
            self.activations[i] = activations[i]
        # batchnorm module for each hidden layer
        self.norm = nn.ModuleList([nn.BatchNorm1d(dim) for dim in layers[1:-1]])
        self.use_norm = batch_norm

    def forward(self, x):
        """
        propagates an activation through the network
        :param x: network input
        :return: network output
        """
        for layer, norm, activation in zip(self.layers[:-1], self.norm, self.activations):
            x = activation(layer(x))
            if self.use_norm:
                x = norm(x)
        x = self.layers[-1](x)
        return x


class EnvironmentModelSeparateReward(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers, activations, batch_norm=True):
        """
        An environment model that models the reward function of the environment with a separate neural nework.
        :param state_dim: dimension of states in the environment
        :param action_dim: dimension of action in the environment
        :param hidden_layers: hidden layers of the nerworks
        :param activations: per layer activations of the networks, if less than layers, defaults will be used
        :param batch_norm: whether to use batch normalization between layers
        """
        super(EnvironmentModelSeparateReward, self).__init__()
        layers_dynamics = [state_dim+action_dim]+hidden_layers+[state_dim]
        layers_reward = [state_dim+action_dim]+hidden_layers+[1]
        self.model_dynamics = NN(layers_dynamics, activations, batch_norm)
        self.model_reward = NN(layers_reward, activations, batch_norm)

    def forward(self, x, catreward=True):
        """
        Propagates an input through the dynamics and reward network
        :param x: the input
        :param catreward: whether to concatenate the outputs of dynamics and reward model such they appear as one
        :return: the output of the networks
        """
        x = torch.squeeze(x)
        if catreward:
            return torch.cat((self.model_dynamics(x), self.model_reward(x)), dim=-1)
        else:
            return self.model_dynamics(x), self.model_reward(x)

    def propagate(self, state, action):
        """
        See documentation of forward. Concatenates state and action before forwarding.
        :param state: input state
        :param action: input action
        :return: the next state
        """
        inp = torch.cat((state, action), dim=-1)  # use negative dim so we can input batches aswell as single values
        output = self.forward(inp)
        output = torch.squeeze(output)
        if output.dim() == 1:
            return output[:-1], output[-1]
        else:
            return output[:, :-1], output[:, -1]


class EnvironmentModel(NN):

    def __init__(self, state_dim, action_dim, hidden_layers, activations, batch_norm=True, predicts_delta=True):
        """
        A neural network parameterized to model an OpenAI Gym environments state transition.
        :param state_dim: dimension of the environments state space
        :param action_dim: dimension of the environments action space
        :param hidden_layers: hidden_layers is a list defining the number of hidden nodes per layer
        :param activations: list of activation functions for each layer
        :param batch_norm: bool whether to use batch normalization
        :param predicts_delta: bool whether the model should predict the next state S' directly(False) or by
         predicting the delta(True) to the next state. dS = S' - S
        """
        layers = [state_dim+action_dim]+hidden_layers+[state_dim]
        super(EnvironmentModel, self).__init__(layers, activations)
        self.predicts_delta = predicts_delta
        self.probabilistic = False

    def propagate(self, state, action):
        """
        Calculates the next state with the model.
        :param state: input state
        :param action: input action
        :return: the next state
        """
        input = torch.cat((state, action), dim=-1) # use negative dim so we can input batches aswell as single values
        # input = state
        output = self.forward(input)
        if self.predicts_delta:
            # in this case we obtain the next state by adding the delta to starting state
            output = state + output
        return output


class ProbabilisticEnvironmentModel(NN):
    # the variance bounds were taken from the handful of trials source code
    def __init__(self, state_dim, action_dim, hidden_layers, activations, variance_bound=[1.e-5, 0.5], batch_norm=True, predicts_delta=True, propagate_probabilistic=True):
        """
        A neural network parameterized to model an OpenAI Gym environemnt.
        Also outputs the diagonal covariance of the prediction.
        This means the output layer will be of format [mean:variance].
        :param state_dim: dimension of the environments state space
        :param action_dim: dimension of the environments action space
        :param hidden_layers: hidden_layers is a list defining the number of hidden nodes per layer
        :param activations: list of activation functions for each layer
        :param batch_norm: bool whether to use batch normalization
        :param predicts_delta: bool whether the model should predict the next state S' directly(False) or by
         predicting the delta(True) to the next state. dS = S' - S
        """
        self.mean_dim = state_dim
        self.variance_bound = variance_bound
        self.predicts_delta = predicts_delta
        layers = [state_dim+action_dim]+hidden_layers+[state_dim*2] # *2 for variance
        super(ProbabilisticEnvironmentModel, self).__init__(layers, activations, batch_norm)
        self.softplus = torch.nn.Softplus()
        self.propagate_probabilistic = propagate_probabilistic
        self.probabilistic = True
        self.max_logvar = np.log(variance_bound[1])
        self.min_logvar = np.log(variance_bound[0])

    def propagate(self, state, action):
        """
        Predicts the next state with the model.
        :param state: input state
        :param action: input action
        :return: the next state
        """
        mean, var = self.propagate_dist(state, action)
        if self.propagate_probabilistic:
            covariance_matrices = torch.stack([torch.diagflat(v) for v in var])
            # create a batched distribution
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrices)
            # it would also be feasible to only use the means as output
            # this would mean our NN is deterministic but trained using the variance
            output = dist.sample()
        else:
            output = mean
        return output

    def propagate_dist(self, state, action):
        """
        Predicts the next state with the model and also outputs the variance of the prediction.
        :param state:
        :param action:
        :return:
        """
        inp = torch.cat((state, action), dim=-1)  # use negative dim so we can input batches aswell as single values
        output = self.forward(inp)
        # probabilistic NN outputs mean var
        mean, var = torch.chunk(output, 2, dim=-1)
        # clip variance
        log_var = torch.log(var)
        log_var = self.max_logvar - self.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + self.softplus(log_var - self.min_logvar)
        var = torch.exp(log_var)
        # create a distribution for each pair of mean and variance
        if self.predicts_delta:
            # in this case we obtain the next state by adding the delta to starting state
            mean = state + mean
        return mean, var


class EnsembleEnvironmentModel:
    """
    Multiple models combined into an ensemble.
    The input is propagated with each model.
    """

    def __init__(self, models):
        """
        :param models: the models of the ensemble
        """
        self.models = nn.ModuleList(models)

    def is_ensemble(self):
        return True

    def propagate(self, s, a):
        """
        predicts the next states witha all models from the environment and returns a list of predictions
        :param s: input state
        :param a: input actions
        :return: list of next state predictions
        """
        return [m.propagate(s, a) for m in self.models]

    def forward(self, x):
        """
        :param x: input to the models
        :return: outputs of all models
        """
        return [m(x) for m in self.models]


class ModelTrainer:

    def __init__(self, model, loss_func=nn.MSELoss(), optimizer=torch.optim.Adam, weight_decay=0, lr=1e-2, lr_min=1e-5,
                 lr_decay=1., batch_size=50, epochs=1, logging=False, plotting=False):
        """
        ModelTrainer optimized the parameters of a model.
        :param model: the model to optimize
        :param loss_func: the loss function that should be minimized
        :param optimizer: a function/constructor that returns a torch.optim optimizer
        :param weight_decay: a list or value specifying the weight decay for each layer
        :param lr: learn rate for the optimizer
        :param lr_decay: learn rate decay for the optimizer
        :param lr_min: minimum learnrate
        :param batch_size: the number of data points to evaluate the model on before changing parameters
        :param epochs: how often the model is trained with the same data
        """
        self.model = model
        self.logging = logging
        self.plotting = plotting
        self.lossFunc = loss_func
        params = []
        for i, layer in enumerate(model.layers):
            param = {'params':layer.parameters()}
            if isinstance(weight_decay, list) and i < len(weight_decay):
                param['weight_decay'] = weight_decay[i]
            if isinstance(lr, list) and i < len(lr):
                param['lr'] = lr[i]
            params.append(param)

        self.optimizer = optimizer(params, lr=lr[-1] if isinstance(lr, list) else lr, weight_decay=0 if isinstance(weight_decay, list) else weight_decay)
        # self.optimizer = optimizer(model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_min = lr_min
        if lr_decay != 1.:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.scheduler = None

    def train(self, inputs, targets, epochs=0):
        """
        train updates the parameters of the model to minimize the loss function on inputs and targets
        the train data is shuffled for each epoch.
        :param inputs: the training inputs
        :param targets: the training targets
        :param epochs: how many times to iterate over the inputs and targets, defaults to self.epochs
        """
        if epochs == 0:
            epochs = self.epochs

        self.model.train()
        mse_loss_func = torch.nn.MSELoss()
        losses = []
        mse_losses = []

        for e in range(epochs):
            if self.logging:
                print("Epoch: {}/{}".format(e, epochs))
            permutation = torch.randperm(len(inputs))
            for batch_in, batch_t in zip(torch.split(inputs[permutation], self.batch_size), torch.split(targets[permutation], self.batch_size)):
                if len(batch_in) < 2:
                    continue
                batch_pred = self.model(batch_in)
                loss = self.lossFunc(batch_pred.float(), batch_t.float())
                if self.model.probabilistic:
                    mse_losses.append(mse_loss_func(torch.chunk(batch_pred, 2, dim=-1)[0], batch_t.float()).item())
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.scheduler is not None and self.scheduler.get_lr()[0] > self.lr_min:
            self.scheduler.step()
            print("LR: ", self.scheduler.get_lr())

        if self.logging:
            print("Mean loss: ", np.mean(losses))
        if self.plotting:
            plt.figure(figsize=(5,5))
            plt.plot(losses, "r-")
            if self.model.probabilistic:
                plt.plot(mse_losses, "b-")
            plt.show()


class EnsembleTrainer:

    def __init__(self, ensemble, trainers):
        """
        EnsembleTrainer is composed of multiple trainers, one for each model in the ensemble.
        :param ensemble: the ensemble model to train, instance of EnsembleEnvironmentModel
        :param trainers: a list of model trainers for the ensemble models
        """
        if not len(trainers) == len(ensemble.models):
            raise ValueError("As many trainers as models needed!")
        self.trainers = trainers

    def train(self, inputs, targets):
        """
        trains all models of the ensemble with the given inputs and targets.
        The individual models might be trained with different subsets of the data.
        :param inputs: the input data
        :param targets: target data
        """
        for trainer in self.trainers:
            samples = np.random.uniform(0, len(inputs), size=len(inputs))
            trainer.train(inputs[samples], targets[samples])

class IdleTrainer:
    """
    This is a trainer for models that cannot be trained
    """
    def __init__(self):
        pass

    def train(self, inputs, targets, epochs=-1):
        pass

# taken from git.ias.informatik.tu-darmstadt.de/quanser/clients
class PerfectDynamicsModelCartpole(nn.Module):

    def __init__(self, long=False):
        super(PerfectDynamicsModelCartpole, self).__init__()
        self.dt = 0.002

        self._g = 9.81              # Gravitational acceleration [m/s^2]
        self._eta_m = 1.            # Motor efficiency  []
        self._eta_g = 1.            # Planetary Gearbox Efficiency []
        self._Kg = 3.71             # Planetary Gearbox Gear Ratio
        self._Jm = 3.9E-7           # Rotor inertia [kg.m^2]
        self._r_mp = 6.35E-3        # Motor Pinion radius [m]
        self._Rm = 2.6              # Motor armature Resistance [Ohm]
        self._Kt = .00767           # Motor Torque Constant [N.zz/A]
        self._Km = .00767           # Motor Torque Constant [N.zz/A]
        self._mc = 0.38             # Mass of the cart [kg]

        if long:
            self._mp = 0.23         # Mass of the pole [kg]
            self._pl = 0.641 / 2.   # Half of the pole length [m]

            self._Beq = 5.4        # Equivalent Viscous damping Coefficient 5.4
            self._Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.3
            self.scale = np.array([0.45,1.])
        else:
            self._mp = 0.127        # Mass of the pole [kg]
            self._pl = 0.3365 / 2.  # Half of the pole length [m]
            self._Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
            self._Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.5
            self.scale = np.array([1.,1.])

        # Compute Inertia:
        self._Jp = self._pl**2 * self._mp/3.   # Pole inertia [kg.m^2]
        self._Jeq = self._mc + (self._eta_g * self._Kg**2 * self._Jm)/(self._r_mp**2)

    def forward(self, input):#state, V_m):
        input = input.squeeze()
        input = input.detach().numpy()

        x = input[:,0]
        theta_sin = input[:,1]
        theta_cos =  input[:,2]
        x_dot = input[:,3]
        theta_dot = input[:,4]
        action = input[:,5]
        theta = np.arctan2(theta_cos, theta_sin)

        F = self.gain * (self._eta_g * self._Kg * self._eta_m * self._Kt) / (self._Rm * self._r_mp) * (- self._Kg * self._Km * x_dot / self._r_mp + self._eta_m * action)

        A = np.array([
            [[theta_cos, self._pl],
            [self._mp + self._mc, self._pl * self._mp * theta_cos]]
            for theta_cos in theta_cos])

        b = np.array([
            [-self._g * theta_sin - self._Bp * theta_dot,
            F + self._mp * self._pl * theta_dot**2 * theta_sin - self._Beq*x_dot]
            for theta_sin, theta_dot, F, x_dot in zip(theta_sin, theta_dot, F, x_dot)])

        solution = np.linalg.solve(A, b) * self.scale
        x_ddot = solution[:,0]
        theta_ddot = solution[:,1]
        theta_dot_new = theta_dot + theta_ddot * self.dt
        x_dot_new = x_dot + x_ddot * self.dt
        theta_new = theta + theta_dot * self.dt
        x_new = x + x_dot * self.dt
        output = np.transpose([x_new, np.sin(theta_new), np.cos(theta_new), x_dot_new, theta_dot_new])
        return torch.tensor(output)



    def propagate(self, state, action):
        return self.forward(torch.cat((state, action), dim=-1))

# taken from openAI Pendulum-v0 environment
class PerfectDynamicsModelPendulum(nn.Module):
        """
        A perfect analytically implemented model of the cartpole, that matches the interface of the approximate models.
        """
        def __init__(self):
            super(PerfectDynamicsModelPendulum, self).__init__()

        def eval_print(self):
            pass

        def forward(self, input):
            """
            Calculates next state for given state and action pair
            param input: consisting of state action pair in an array
            return: next state
            """
            g = 10.
            m = 1.
            l = 1.
            dt = .05

            torch.squeeze(input)
            if input.dim() == 1:  # we ave a single state action pair
                state = input[:3]
                action = input[3:]
                theta = angle_from_sincos(state[:1], state[1:2])
                theta_dot = state[2:]

                # cost = _angle_normalize(theta) ** 2 + .1 * theta_dot ** 2 + .001 * u ** 2
            else:  # we have a batch
                state = input[:,:3]
                action = input[:,3:]
                theta = angle_from_sincos(state[:, 0], state[:, 1])
                theta_dot = state[:, 2]

            u = torch.clamp(action.squeeze(), -2, 2)

            new_theta_dot = theta_dot \
                            + (-3 * g / (2 * l) * torch.sin(theta + np.pi) \
                               + 3. / (m * l ** 2) * u) * dt
            new_theta = theta + new_theta_dot * dt
            new_theta_dot = torch.clamp(new_theta_dot, -8, 8)

            return torch.stack((torch.cos(new_theta), torch.sin(new_theta), new_theta_dot), dim=-1)

        def propagate(self, state, action):
            return self.forward(torch.cat((state, action), dim=-1))

perfect_models = {
    "Pendulum-v0": PerfectDynamicsModelPendulum(),
    "CartpoleStabShort-v0": PerfectDynamicsModelCartpole(),
    "CartpoleSwingShort-v0": PerfectDynamicsModelCartpole()
}