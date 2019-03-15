# DDPG

This implementation follows the [original paper](http://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models.pdf) with slightly different hyperparameters as [described here](/src/config/DDPG/Readme.md).

To create a new MPC instance

    # create a model
    model = EnvironmentModel(state_dim, action_dim, [100,100], [], predicts_delta=True,batch_norm=True).to('cpu')
    # select a loss function that works on our model
    loss = torch.nn.MSELoss()
    # create a trainer for the model
    trainer = ModelTrainer(model, loss_func=loss, weight_decay=[1.e-5, 1.e-5],epochs=5, lr=1.e-3, lr_decay=1., batch_size=25)
    # plug evey thing into mpc
    mpc = MPC(gym.make("Pendulum-v0"), rewards["Pendulum-v0"], model, trainer, trial_horizon=1000,
              device="cpu", warmup_trials=5, learning_trials=30,
              predict_horizon=12, render=5)
    # start training
    mpc.sim_train()
    
MPC is not built to learn rewards. This means that we can only learn environments for which a reward function was implemented and registered in `reward.py`. The key of a reward function in the rewards dict should equal the name of the environment the reward function should be used for. 

MPC currently does not support saving and loading of learnt models.