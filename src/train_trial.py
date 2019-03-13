import csv
import os
from src.config.AlgorithmParser import AlgorithmParser


def write_config(configuration, configfile):
    """
    Write your config in specified in configfile for every training session
    :param configuration: a dictionary containing your configuration for a session
    :param configfile: file specified to save in
    :return:
    """
    with open(configfile, 'w+') as csvfile:
        w = csv.DictWriter(csvfile, configuration.keys(), delimiter=';')
        w.writeheader()
        w.writerow(configuration)


def train(run_configs, algorithm, outdir, mode):
    """
    Training mode selected
    :param run_configs: .csv file to read from that specifies training sessions
    :param algorithm: algorithm to use as a string
    :param outdir: output directory to use for this training session
    :param mode: the mode that has been chosen either "sim" or "rr"

    :return:
    """

    algorithm_parser = AlgorithmParser(run_configs=run_configs, algorithm= algorithm)

    # with this algorithm class we can start training
    algorithm_class = algorithm_parser.algorithm_class

    for conf in algorithm_parser.run_configs:

        algorithm = algorithm_class(conf)

        run_out_dir = os.path.join(outdir, "{}_{}".format(conf["run_id"], conf["env"]))

        if not os.path.exists(run_out_dir):
            os.makedirs(run_out_dir)
        else:
            raise Exception("output directory '/{}' should not exist or be empty.".format(run_out_dir))

        # so you remember which parameters you used before :)
        write_config(conf, os.path.join(run_out_dir, "parameters.csv"))

        if mode == 'rr':
            # every developer decides for himself what he wants to do with his training result (whatever this is)
            return_train = algorithm.train_rr()
        if mode == 'sim':
            return_train = algorithm.train_sim()

        algorithm_parser.handle_result(return_train, run_out_dir)


def trial(algorithm, policy, outdir, mode, episodes):
    """
    Trial mode of your favorite algorithms
    :param algorithm: string of algorithm that is used
    :param policy: path to policy folder containing parameters.csv and policy
    :param outdir: output directory used to save your results (if even used)
    :param mode: either simulated or real environment ('rr', 'sim')
    :param episodes: number of episodes your policy will be tested
    :return:
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        print("Folder {} will be overwritten with possible results".format(outdir))

    run_configs = policy+"/parameters.csv"
    algorithm_parser = AlgorithmParser(run_configs=run_configs, algorithm= algorithm)

    # with this algorithm class we can start training
    algorithm_class = algorithm_parser.algorithm_class

    # since there is only config used in a
    conf = algorithm_parser.run_configs[0]
    algorithm = algorithm_class(conf)

    # every algorithm needs a load_model function
    # this will always have a side-effect on the intern policy the algorithm is using
    algorithm.load_model(policy+"/policy")

    if mode == 'rr':
        # every developer decides for himself what he wants to do with his training result (whatever this is)
        return_trial = algorithm.trial_rr(episodes)
    if mode == 'sim':
        return_trial = algorithm.trial_sim(episodes)

    algorithm_parser.handle_result(result=return_trial, outdir=outdir)



