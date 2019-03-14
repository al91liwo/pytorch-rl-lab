import gym
import torch.nn
import ast
import os
import matplotlib.pyplot as plt
# this is the validate function to validate any config under the developers layout constraint
from src.algorithm.MPC.model import ProbabilisticEnvironmentModel, NegLogLikelihood, ModelTrainer, EnvironmentModel, perfect_models, IdleTrainer
from src.algorithm.MPC.reward import rewards_t as rewards
from src.utility.util import validate_config
# import the developers algorithm here
import os.path
from src.algorithm.MPC.mpc import MPC


def layout():
    """
    Layout of a developers algorithm, the specified parameters are used as default values!
    :return: layout as a dict
    """
    layout_dict = {
        #every layout needs a run_id param
        "run_id": 0,
        "env": 0,
        "dirname": "out",
        "batch_size": 64,
        "is_quanser_env": True,
        "gamma": .99,
        "batch_norm": True,
        "steps": 100000,
        "warmup_samples": 1000,
        "noise_decay": 0.9,
        "transform": lambda x:x,
        "lr": 1e-3,
        "lr_decay": 1.0,
        "lr_min":1.e-8,
        "trial_horizon": 1000,
        "layers":[10, 10, 10],
        "predicts_delta": True,
        "weight_decay": 0,
        "model": "perfect",
        "propagate_probabilistic":False,
        "device": "cpu"
    }
    return layout_dict

lossFunctions = {
    "mse": torch.nn.MSELoss,
    "nll": NegLogLikelihood
}

def instance_from_config(config):
    """
    The developer creates a instance of his algorithm and returns it to the config parser
    :param config: the config to load (does not have to be fully specified)
    :return: an instance of the developers algorithm (DDPG example)
    """
    layout_dict = layout()
    validate_config(config, layout_dict)
    print(config)
    for key in config:
        try:
            config[key] = ast.literal_eval(config[key])
        except ValueError:
            pass

    # merging config into layout, EVERY layout needs a "run_id" variable
    layout_dict.update(config)

    if "run_id" not in layout_dict.keys():
        raise Exception('Every config needs a "run_id"')

    env = gym.make(layout_dict["env"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    if layout_dict["model"] == "pnn":
        model = ProbabilisticEnvironmentModel(state_dim, action_dim, layout_dict["layers"], [],
                                              predicts_delta=layout_dict["predicts_delta"],
                                              propagate_probabilistic=layout_dict["propagate_probabilistic"]).to(
            layout_dict["device"])
        loss = NegLogLikelihood()
    elif layout_dict["model"] == "nn":
        model = EnvironmentModel(state_dim, action_dim, layout_dict["layers"], [], predicts_delta=True, batch_norm=layout_dict["batch_norm"]).to(layout_dict["device"])
    elif layout_dict["model"] == "perfect":
        if layout_dict["env"] in perfect_models:
            model = perfect_models[layout_dict["env"]]
        else:
            raise NotImplementedError("A perfect model for environment {} does not exist!".format(layout_dict["env"]))

    if layout_dict["model"] == "perfect":
        trainer = IdleTrainer()
    else:
        trainer = ModelTrainer(model, lossFunc=loss, weight_decay=layout_dict["weight_decay"], epochs=layout_dict["epoch"], lr=layout_dict["lr"], lr_decay=layout_dict["lr_deacy"], lr_min=layout_dict["lr_min"], batch_size=layout_dict["batch_size"], logging=layout_dict["logging"], plotting=False)

    if not layout_dict["env"] in rewards:
        raise NotImplementedError("A reward function for the environment {} does not exist!".format(layout["env"]))

    mpc = MPC(env, rewards[layout_dict["env"]], model, trainer, trial_horizon=layout_dict["trial_horizon"], device=layout_dict["device"], warmup_trials=layout_dict["warump_trials"], trials=layout_dict["trials"],
              predict_horizon=layout_dict["predict_horizon"], cem_samples=layout_dict["cem_samples"], nelite=layout_dict["elites"])

    return mpc


def result_handler(result, outdir):
    """
    The developer handles the result of one training session of his specified algorithm right here
    :param result: the result of one training session
    :param outdir: directory you can use to do whatever with (e.g. save plot into directory), for every training session
    there will be a new directory given
    """
    with open(os.path.join(outdir, 'rewarddata'), 'w') as fout:
        fout.write(','.join([str(r) for r in result]))

    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.plot(result)
    plt.savefig(os.path.join(outdir, "rewardplot.png"))
    print(result)



