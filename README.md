# pytorch-rl-lab
DDPG and MPC implementation of Group 06 Frederik Wegner and Alexander Lind

## Installation Guide

This guide assumes you are working under Ubuntu 16.04

1. Make sure you installed all dependencies of the dependencies.yaml.
   Create a virtual environment, activate it, and update it.
   You can also use an Anaconda virtual environment.

        python3.6 -m venv venv3
        source venv3/bin/activate
        pip3 install -U pip setuptools
   
        conda install dependencies.yaml
   More details on how to activate and use environments can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

2. Clone this repository into some folder to use the quanser_environment

        git clone --recursive git@github.com:al91liwo/pytorch-rl-lab.git
            or
        git clone --recursive https://github.com/al91liwo/pytorch-rl-lab.git

3. Make sure you have Python >= 3.5.3 on your system. If that is not the case,
   install Python3.6

        sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install python3.6
        sudo apt-get install python3.6-venv
    
    More details on that can be found [here](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get).

4. Install the `quanser_robots` package

        cd clients
        pip3 install -e .

5. Check that everything works correctly by running the code snippet
   from the [example quanser_environment](https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/Readme.md) and [pytorch-rl-lab example](/src/config/example/Readme.md).
   

## Getting started

You have the choice between training or doing trials of an specified algorithm of our [algorithms](src/algorithm/Readme.md).
It is possbile to train or test the algorithms on the real environment ('rr') or the simulated environment ('sim')

Generally we have specified two different commands

1. training

        python main.py YOUR_ALGORITHM {sim/rr} train hyperparameters outdir
2. trial

        python main.py YOUR_ALGORITHM {sim/rr} trial hyperparameters

where you always can choose between `{sim/rr}` the simulated environment `sim` or the real environment `rr`.

## Example

For example you can train the algorithm [DDPG](src/algorithm/DDPG/Readme.md) with given hyperparameters as a .csv file. For example [parameters.csv](parameters.csv)

| run_id        | env                  | steps  | batch_size | buffer_size | warmup_samples | actor_lr | critic_lr | actor_hidden_layers | critic_hidden_layers | tau  | noise_decay | lr_decay | lr_min     | 
|---------------|----------------------|--------|------------|-------------|----------------|----------|-----------|---------------------|----------------------|------|-------------|----------|------------| 
| CartpoleTrial | CartpoleStabShort-v0 | 100000 | 64         | 1000000     | 100            | 0.001    | 0.001     | [100,150]           | [100,150]            | 0.01 | 0.99        | 1.       | 0.00000001 | 

Execute this command to obtain results:

    python main.py ddpg sim train parameters.csv out
    

`out` specifies the directory where the output result will be saved (this is strictly specified by the developer) for more information take a look at [config readme](src/config/Readme.md)

`train` the command to train the specified algorithm under given hyperparameters (the parameters.csv) file