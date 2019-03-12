import csv
import os
from src.config.ConfigParser import ConfigParser


def write_config(configuration, configfile):
    with open(configfile, 'w+') as csvfile:
        w = csv.DictWriter(csvfile, configuration.keys(), delimiter=';')
        w.writeheader()
        w.writerow(configuration)


def train(run_configs, algorithm, outdir):
    """
    Training mode selected
    :param run_configs: .csv file to read from that specifies training sessions
    :param algorithm: algorithm to use as a string
    :param outdir: output directory to use for this training session
    :return:
    """

    config = ConfigParser(run_configs=run_configs, algorithm= algorithm)

    # with this algorithm class we can start training
    algorithm_class = config.algorithm_class

    for conf in config.run_configs:

        algorithm = algorithm_class(conf)

        run_outdir = os.path.join(outdir, "{}_{}".format(conf["run_id"], algorithm.env))

        if not os.path.exists(run_outdir):
            os.makedirs(run_outdir)
        else:
            raise Exception("output directory '/{}' should not exist or be empty.".format(run_outdir))

        # so you remember which parameters you used before :)
        write_config(conf, os.path.join(run_outdir, "parameters.csv"))

        # every developer decides for himself what he wants to do with his training result (whatever this is)
        train_return = algorithm.train()

        config.handle_result(train_return, run_outdir)




