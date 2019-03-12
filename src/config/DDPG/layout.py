# this is the validate function to validate any config under the developers layout constraint
from src.util import validate_config
# import the developers algorithm here
from src.algorithm.DDPG.DDPG import DDPG


def layout():
    """
    Layout of a developers algorithm, the specified parameters are used as default values!
    :return: layout as a dict
    """
    layout_dict = {
        "env": 0,
        "action_space_limits": 0,
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
    }
    return layout_dict


def instance_from_config(config):
    """
    The developer creates a instance of his algorithm and returns it to the config parser
    :param config: the config to load (does not have to be fully specified)
    :return: an instance of the developers algorithm (DDPG example)
    """
    layout_dict = layout()
    validate_config(config, layout_dict)
    # merging config into layout, EVERY layout needs a "run_id" variable
    layout_dict.update(config)

    if not "run_id" in layout_dict.keys():
        raise Exception('Every config needs a "run_id"')

    return DDPG(env=layout_dict["env"], action_space_limits=layout_dict["action_space_limits"],
                buffer_size=layout_dict["buffer_size"], batch_size=layout_dict["batch_size"],
                is_quanser_env=layout_dict["is_quanser_env"], gamma=layout_dict["gamma"],
                tau=layout_dict["tau"], steps=layout_dict["steps"], warmup_samples=layout_dict["warmup_samples"],
                noise_decay=layout_dict["noise_decay"], transform=layout_dict["transform"],
                actor_lr=layout_dict["actor_lr"], critic_lr=layout_dict["critic_lr"],
                lr_decay=layout_dict["lr_decay"], lr_min=layout_dict["lr_min"],
                trial_horizon=layout_dict["trial_horizon"], actor_hidden_layers=layout_dict["actor_hidden_layers"],
                critic_hidden_layers=layout_dict["critic_hidden_layers"], device=layout_dict["device"])

def result_handler(result, outdir):
    """
    The developer handles the result of one training session of his specified algorithm right here
    :param result: the result of one training session
    :param outdir: directory you can use to do whatever with (e.g. save plot into directory), for every training session
    there will be a new directory given
    """


