# Hyperparameters

        "run_id": 0,
        "env": 0,
        "action_space_limits": ([-10.], [10.]),
        "buffer_size": 10000,
        "batch_size": 64,
        "is_quanser_env": True,
        "gamma": .99,
        "tau": 1e-2,
        "steps": 100000,
        "warmup_samples": 1000,
        "noise_decay": 0.9,
        "transform": lambda x:x,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "lr_decay": 1.0,
        "lr_min":1.e-7,
        "trial_horizon": 5000,
        "actor_hidden_layers":[10, 10, 10],
        "critic_hidden_layers":[10, 10, 10],
        "device": "cpu"
        
These will be set in every run, either a trial or training session!

`run_id` every algorithm should specify a run_id to obtain a structured order

`env` the environment which is used in this run (e.g. "Pendulum-v0")

`action_space_limits` limits the actions to select between two values, given in a tuple

`buffer_size` the buffer size taken of a [replay buffer](../../utility/ReplayBuffer.py)

`batch_size` size of batches that are sampled from a replay buffer, to train the [actor](../../algorithm/DDPG/ActorNetwork.py) and [critic network](../../algorithm/DDPG/CriticNetwork.py) with

`is_quanser_env` either if the used `env` is a [quanser_robots](https://git.ias.informatik.tu-darmstadt.de/quanser/clients) environment (True) or not (False)

`gamma` used as a weight factor of the critic's prediction

`tau` soft update for the weights from the source to target network

`steps` that will be taken in this training session

`warmup_samples` number of samples generated before starting the training

`noise_decay` the gaussian noise will be reduced by this factor in every episode

`transform` user specific transformation function (for dimensionality reduction for example)

`actor_lr` adam optimizer learning rate of your actor networks

`critic_lr` adam optimizer learning rate of your critic networks

`lr_decay` decay of adam optimizers learning rates

`lr_min` your optimizer learning rate cant drop this value

`trial_horizon` in between training episodes number of trials to test your new policy

`actor_hidden_layers` a numeric list of layers between input and output layer that will be used in your actor networks

`critic_hidden_layers` a numeric list of layers between input and output layer that will be used in your critic networks

`device` either cpu or cuda (faster learning), tensors will be calculated on specified device