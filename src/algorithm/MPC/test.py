import src.algorithm.MPC.control as control
import torch
import src.algorithm.MPC.model as mdl
import numpy as np
import math
import matplotlib.pyplot as plt


def rosebrock(x):
    return (1 - x[0])**2 + 100(x[1] - x[0]**2)**2


def test_cem_optimizer_2dparabel():
    f = lambda x: torch.tensor([torch.sum(-x**2) for x in x])
    start = torch.tensor([[3., 3.], [3., 3.]])
    minimum = control.cem_optimize(start, torch.tensor([[5.,5.], [5.,5.]]), f, (2, 2), 20)
    err = torch.mean(minimum - torch.tensor([[0., 0.]]))
    print("Approximated 2dparabel with err ", err)

def test_perfect_dynamics_pendulum():
    """
    test_perfect_dynamics_pendulum tests the torch implementation of the analytical perfect
    pendulum dynamics that also supports batch input against the original numpy implementation.
    Testing is done by creating state trajectory from an action trajectory using both dynamic
    models and then comparing for equality.
    """

    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    def angle_from_sincos(cos, sin):
        """
        Calculates the angle from sin and cos
        :param sin: the sin of the angle
        :param cos: the cos of the angle
        :return: the angle in radiants
        """
        sin_angle = math.asin(sin)
        cos_angle = math.acos(cos)
        if sin_angle == 0:
            return cos_angle
        return cos_angle * sin_angle / abs(sin_angle)

    def step(state, u):
        max_speed = 8.
        max_torque = 2.
        g = 10.
        m = 1.
        l = 1.
        dt = .05


        th = angle_from_sincos(state[0], state[1]) # th := theta
        thdot = state[2]

        u = np.clip(u, -max_torque, max_torque)
        last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -max_speed, max_speed) #pylint: disable=E1111

        return np.array([np.cos(newth), np.sin(newth), newthdot])



    perfect_dyn_torch = mdl.perfect["Pendulum-v0"]()
    perfect_dyn_original = step

    max_error = 1.e-6

    actions = np.random.random_sample(20) * 4. - 2.
    start_state = np.random.random(3) * np.array([2., 2., 16.]) - np.array([1., 1., 8.])
    curr_state = start_state
    for a in actions:
        next_state = perfect_dyn_original(curr_state, a.item())
        curr_state_t = torch.tensor(curr_state).float()
        a_t = torch.tensor([a])
        next_state_torch = perfect_dyn_torch.propagate(curr_state_t, a_t).detach().numpy()
        curr_state_t_batch = curr_state_t.repeat(5, 1)
        a_t_batch = a_t.repeat(5, 1)
        next_state_torch_batch = perfect_dyn_torch.propagate(curr_state_t_batch, a_t_batch)

        if np.any(np.abs(next_state - next_state_torch) > max_error):
            return "failed. Next state %s, differed more than %f from torch model %s" % (next_state, max_error, next_state_torch)

        if torch.any(torch.abs(next_state_torch_batch - torch.tensor(next_state, dtype=torch.float32).repeat(5,1)) > max_error):
            return "batch failed. Next state %s, differed more than %f from torch model %s" % (next_state_torch_batch, max_error, torch.tensor(next_state).repeat(5,1))

        curr_state = next_state

def test_probmodel(f, X_train, Y_train, trials):
    """
    we simulate the task of learning an environment by increasing the amount of samples of the toy function over time
    :return:
    """
    model = mdl.ProbabilisticEnvironmentModel(1, 0, [100,100,200], [], predicts_delta=False)
    trainer = mdl.ModelTrainer(model, lossFunc=mdl.NegLogLikelihood(), epochs=5, lr=1.e-3, weight_decay=1e-4, batch_size=25, logging=True)

    X = np.linspace(min(X_train)-1., max(X_train)+1., 400)
    Y = f(X)

    mean, var = model.propagate_dist(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
    mean = mean.squeeze().detach().numpy()
    var = var.squeeze().detach().numpy()

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    truthline, = ax.plot(X, Y, 'g')
    traindata, = ax.plot(X_train, Y_train, 'o')
    approx_mean, = ax.plot(X, mean, '-')
    ax.fill_between(X, mean - var, mean + var, alpha=0.2, color='k')
    plt.draw()

    for _ in range(trials):
        trainer.train(torch.tensor(X_train).float().unsqueeze(1), torch.tensor(Y_train).float().unsqueeze(1))

        mean, var = model.propagate_dist(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
        mean = mean.squeeze().detach().numpy()
        var = var.squeeze().detach().numpy()

        traindata.set_xdata(X_train)
        traindata.set_ydata(Y_train)
        approx_mean.set_ydata(mean)
        ax.collections.clear()
        ax.fill_between(X, mean - var, mean + var, color='k', alpha=0.2)
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    # plt.show()


def test_detmodel(f, X_train, Y_train, trials):
    """
    we simulate the task of learning an environment by increasing the amount of samples of the toy function over time
    :return:
    """
    model = mdl.EnvironmentModel(1, 0, [20,20, 20, 20, 20, 20], [], predicts_delta=False)
    trainer = mdl.ModelTrainer(model, epochs=5, lr=1.e-3, weight_decay=1e-4, batch_size=25, logging=True)

    X = np.linspace(-4., 4., 400)
    Y = f(X)

    mean = model.propagate(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
    mean = mean.squeeze().detach().numpy()

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    truthline, = ax.plot(X, Y, 'g')
    traindata, = ax.plot(X_train, Y_train, 'o')
    approx_mean, = ax.plot(X, mean, '-')
    plt.draw()

    for _ in range(trials):
        trainer.train(torch.tensor(X_train).float().unsqueeze(1), torch.tensor(Y_train).float().unsqueeze(1))

        mean = model.propagate(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
        mean = mean.squeeze().detach().numpy()

        traindata.set_xdata(X_train)
        traindata.set_ydata(Y_train)
        approx_mean.set_ydata(mean)
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    # plt.show()


def test_detensemblemodel(f, X_train, Y_train, trials):
    """
    we simulate the task of learning an environment by increasing the amount of samples of the toy function over time
    :return:
    """
    model = mdl.EnsembleEnvironmentModel([mdl.EnvironmentModel(1, 0, [20,20,20], [], predicts_delta=False) for _ in range(5)])
    trainer = mdl.EnsembleTrainer(model, [mdl.ModelTrainer(m, epochs=10, lr=1.e-3, weight_decay=1e-4, batch_size=25, logging=False) for m in model.models ])

    X = np.linspace(-4., 4., 400)
    Y = f(X)

    means = model.propagate(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
    means = [mean.squeeze().detach().numpy() for mean in means]

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    truthline, = ax.plot(X, Y, 'g')
    traindata, = ax.plot(X_train, Y_train, 'o')
    approx_means = [ax.plot(X, mean, '-')[0] for mean in means]
    plt.draw()

    for _ in range(trials):
        trainer.train(torch.tensor(X_train).float().unsqueeze(1), torch.tensor(Y_train).float().unsqueeze(1))

        means = model.propagate(torch.tensor(X).unsqueeze(1).float(), torch.tensor([]))
        means = [mean.squeeze().detach().numpy() for mean in means]

        traindata.set_xdata(X_train)
        traindata.set_ydata(Y_train)
        for approx_mean, mean in zip(approx_means, means):
            approx_mean.set_ydata(mean)
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    # plt.show()


if __name__=="__main__":
    # test_cem_optimizer_2dparabel()
    # msg = test_perfect_dynamics_pendulum()
    # if msg != None:
    #     print("test_perfect_dynamics  _pendulum"," failed with message: ", msg)

    # test_learn_pendulum_dynamics()

    X_train = np.concatenate((np.random.uniform(-4., -1., 80), np.random.uniform(1.5, 4., 50))) # 100 initial data points
    #
    # f = lambda x: np.power(.2 * (x - 2), 4) + .2 - np.power(.5 * (x), 2)
    # test_probmodel(f, X_train, f(X_train) + np.random.normal(0,0.1, len(X_train)), 50)
    # test_detmodel(f, X_train, f(X_train) + np.random.normal(0,0.1, len(X_train)), 50)
    #
    f = lambda x: np.sin(0.75*x) + .2
    test_probmodel(f, X_train, f(X_train) + np.random.normal(0,0.05, len(X_train)), 50)
    test_detmodel(f, X_train, f(X_train) + np.random.normal(0,0.05, len(X_train)), 50)
    test_detensemblemodel(f, X_train, f(X_train) + np.random.normal(0,0.05, len(X_train)), 50)

    # f = lambda x: np.cos(x)
    # test_probmodel(f, X_train, f(X_train) + np.random.normal(0,0.1, len(X_train)), 50)
    # test_detmodel(f, X_train, f(X_train) + np.random.normal(0,0.1, len(X_train)), 50)
    # test_detensemblemodel(f, X_train, f(X_train) + np.random.normal(0,0.1, len(X_train)), 50)