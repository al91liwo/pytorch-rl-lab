from src.util import validate_config
import csv
import src.algorithm
import src.config

def getAlgorithmConfigLayout(algorithm):
    """
    This right now specifies all available algorithms in this repository,
    will be restructured in the future
    :param algorithm: usable algorithms as a string (follow conventions used)
    :return: config getter of specified algorithm
    """
    configs = {
        "ddpg": "DDPG",
        "mpc": "MPC"
    }
    return configs.get(algorithm, "INVALIDATED")


class AlgorithmParser:

    def __init__(self, run_configs, episodes=0, algorithm="DDPG"):
        """
        Parse the config and starts all training or trial sessions for given configuration
        :param run_configs: .csv file with all training sessions (see Documentation for more information)
        :param algorithm: algorithm name specified by the developers creating rl-algorithms
        """

        self.algorithm = getAlgorithmConfigLayout(algorithm)
        self.algorithm_directory = "src.algorithm."+self.algorithm+"."+self.algorithm
        self.layout_directory = "src.config."+self.algorithm+".layout"

        if self.algorithm == "INVALIDATED":
            raise Exception("given ALGORITHM: "+ self.algorithm + " is not specified in config/ALGORITHM")

        try:
            self.algorithm_layout = __import__(self.layout_directory, fromlist=['config', self.algorithm, 'layout'])
        except Exception as e:
            raise Exception("specified ALGORITHM: " + self.algorithm + " has no layout.py specified in config/" + self.algorithm + "/layout.py")

        try:
            self.layout_dict = self.algorithm_layout.layout()
        except Exception as e:
            raise Exception("specified ALGORITHM: " + self.algorithm +
                            " has no layout function specified in config/" +
                            self.algorithm + "/layout.py")
        try:
            # loading algorithm class from src/algorithm/YOUR_ALGORITHM
            self.algorithm_class = self.algorithm_layout.instance_from_config
        except Exception as e:
            raise Exception("specified ALGORITHM: " + self.algorithm +
                            " has no instance_from_config function specified in config/" +
                            self.algorithm + "/layout.py")
        try:
            self.result_handler = self.algorithm_layout.result_handler
        except Exception as e:
            raise Exception("specified ALGORITHM: " + self.algorithm +
                            " has no result_handler function specified in config/" +
                            self.algorithm + "/layout.py")

        self.run_configs = self.parse_config(run_configs)

        for config in self.run_configs:
            # first we validate every config
            # making sure training sessions will be successfull by clean parameters
            validate_config(config, self.layout_dict)


    def parse_config(self, configfile):
        """
        Given configfile will be parsed
        :param configfile: argument given by user as .csv file
        :return: all configs for every training session (run)
        """
        run_configs = []
        with open(configfile, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                run_configs.append(row)
        return run_configs

    def handle_result(self, result, outdir):
        """
        Every developer can handle his results as he wants
        :param result: the result of a training session
        :param outdir: the output directory that may be used by a developer
        """
        self.result_handler(result, outdir)



