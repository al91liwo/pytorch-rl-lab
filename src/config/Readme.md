# config

Every algorithm has its own folder in this directory.

### adding your own algorithm to this repository

If you want to add your own algorithm to this repository, you need to follow certain steps.

1. Create your algorithm in the [algorithm folder](../algorithm)
    
2. Make sure to add your algorithm to the layout format in [AlgorithmParser.py](AlgorithmParser.py)
    
        configs = {
        "ddpg": "DDPG",
        "mpc": "MPC",
        "your_algorithm": "your_algorithm_folder_name"
        }
   Your algorithm needs to be specified as a folder like `your_algorithm_folder_name` in the [config folder](../config)
   and [algorithm folder](../algorithm)
   
3. Every algorithm in the [config folder](../config) needs to specifiy a layout.py as in [layout.py](DDPG/layout.py)
        
        def layout():
            """
            Layout of a developers algorithm, the specified parameters are used as default values!
            :return: layout as a dict
            """
            layout_dict = {
                #every layout needs a run_id param
                "run_id": 0,
                # here comes your hyperparameters
                "your hyperparameters": ['some', 'cool', 'hyperparameters']
            }
            return layout_dict
        
        
        def instance_from_config(config):
            """
            The developer creates a instance of his algorithm and returns it to the algorithm parser
            :param config: the config to load (needs to fit your needs [containing your hyperparameters])
            :return: an instance of the developers algorithm (DDPG example)
            """
            layout_dict = layout()
            validate_config(config, layout_dict) # validate config should be called here for safety reason [typos]
           
            # load your config into your default hyperparameters
            layout_dict.update(config)
        
            # create an environment to run your algorithm on
            env = gym.make(layout_dict["env"])
            
            # return an instance of your algorithm
            return YOUR_ALGORITHM(run_id=layout_dict["run_id"],
                                  your_hyperparameters=layout_dict["your_hyperparameters"])
        
        def result_handler(result, outdir):
            """
            The developer handles the result of one training session of his specified algorithm right here
            :param result: the result of one training session
            :param outdir: directory you can use to do whatever with (e.g. save plot into directory), for every training session
            there will be a new directory given
            """
            # here you can handle your result of an trial or training session for example
            
            # for example save your results in the output dir
            
            # or simply do nothing if you dont care about results
            return results
    This makes sure that the [train_trial](../train_trial.py) routine works for every algorithm!
    If you get lost look at the [DDPG](../algorithm/DDPG/Readme.md)
   
4. Show some [training results](example/Readme.md) on different environments with given hyperparameters in your pull request.

5. Code clean! Use the [PEP 8 style guide](https://legacy.python.org/dev/peps/pep-0008/).
        
