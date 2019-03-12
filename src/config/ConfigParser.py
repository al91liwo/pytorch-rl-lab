from src.util import validate_config
import csv

def getAlgorithmConfigLayout(algorithm):
    configs = {
        "ddpg": "DDPG",
        "mpc": "MPC"
    }
    return configs.get(algorithm, "INVALIDATED")


class ConfigParser:

    def __init__(self, run_configs, algorithm="DDPG"):
        """
        Parse the config and starts all training sessions for given configuration
        :param run_configs: .csv file with all training sessions (see Documentation for more information)
        :param algorithm: algorithm name specified by the developers creating rl-algorithms
        """

        self.algorithm = getAlgorithmConfigLayout(algorithm)
        self.algorithm_directory = "/algorithm/"+self.algorithm

        if self.algorithm == "INVALIDATED":
            raise Exception("given ALGORITHM: "+ self.algorithm + " is not specified in config/ALGORITHM")

        try:
            # self.layoutDict is a dict of the hyperparameters for given algorithm
            self.layoutDict = __import__(self.algorithm_directory + "layout.py").layout()
        except Exception as e:
            raise Exception("specified ALGORITHM: " + self.algorithm + " has no layout in config/" + self.algorithm + "layout.py")

        try:
            # loading algorithm class from src/algorithm/YOUR_ALGORITHM
            self.algorithm_class = __import__("src/config"+self.algorithm+"/" + self.algorithm + ".py").instance_from_config
        except Exception as e:
            raise Exception(self.algorithm + ".py" + " does not exist in " + self.algorithm_directory)
        try:
            self.result_handler = __import__(self.algorithm_directory+"layout.py").result_handler
        except Exception as e:
            raise Exception("Your algorithm does not define a result_handler function at " +self.algorithm_directory)

        self.run_configs = self.parse_config(run_configs)

        for config in run_configs:
            # first we validate every config
            # making sure training sessions will be successfull by clean parameters
            validate_config(config, self.layoutDict)


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



