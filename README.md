# pytorch-rl-lab
DDPG and MPC implementation of Group 06 Frederik Wegner and Alexander Lind

## Installation Guide

This guide assumes you are working under Ubuntu 16.04

1. Make sure you have Python >= 3.5.3 on your system. If that is not the case, install Python3.6

        sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install python3.6
        sudo apt-get install python3.6-venv
 
2. Clone this repository into some folder:

        git clone git@github.com:al91liwo/pytorch-rl-lab.git
            or
        git clone https://github.com/al91liwo/pytorch-rl-lab.git


4. Create a virtual environment, activate it, and update it. You can also use an Anaconda virtual environment.
 
        python3.6 -m venv venv3
        source venv3/bin/activate
        pip3 install -U pip setuptools
        
5. Install the requirements.

        pip3 install -r requirements.txt
        
5. Check that everything works correctly by running the code snippet
   from the [example quanser_environment](https://git.ias.informatik.tu-darmstadt.de/quanser/clients/blob/master/Readme.md) and [pytorch-rl-lab example](/src/config/example/Readme.md).
   

## Getting started

You can always use the command line to start either a training or trial session with a given algorithm.

    python main.py -h
With python main.py you can specify a algorithm to use. Right now you can use [ddpg](src/algorithm/DDPG/Readme.md) or [mpc](src/algorithm/MPC/Readme.md)

    positional arguments:
    algorithm   algorithm specified in src/algorithm/
    {rr,sim}    choose between simulation or real environment mode

After you've chosen your algorithm, you can either run a session in simulation or real environment mode.

    python main.py ddpg sim -h
Either in simulation or real environment mode you can choose between train or trial mode.

    positional arguments:
    {train,trial}  choose between train or trial
    train        train mode in simulated environment
    trial        trial mode in simulated environment
In train mode you have always to choose a parameters.csv file and a output directory.

    python main.py ddpg sim train -h
You can have a look at the [parameters.csv example](src/config/example/train/parameters.csv) and [a common output directory](/src/config/example/trial)
    
    positional arguments:
    hyperparameters  .csv folder with hyperparameters for specified algorithm
    outdir           output directory of your training data
In trial mode you have always to choose a folder containing a parameters.csv and a policy and the number of episodes to run your policy.

    python main.py ddpg sim trial -h

You can try the [trial example](/src/config/example/trial/Readme.md)

    positional arguments:
    policy      path to your policy
    outdir      save your results in specified directory
    episodes    number of episodes to start your trial in sim mode



## Example

For example you can train the algorithm [DDPG](src/algorithm/DDPG/Readme.md) with given hyperparameters as a .csv file. For example [parameters.csv](parameters.csv)

| run_id        | env                  | steps  | batch_size | buffer_size | warmup_samples | actor_lr | critic_lr | actor_hidden_layers | critic_hidden_layers | tau  | noise_decay | lr_decay | lr_min | batch_norm | trial_horizon | action_space_limits | dirname                                | 
|---------------|----------------------|--------|------------|-------------|----------------|----------|-----------|---------------------|----------------------|------|-------------|----------|--------|------------|---------------|---------------------|----------------------------------------| 
| CartpoleTrial | CartpoleStabShort-v0 | 300000 | 64         | 1000000     | 20000          | 0.001    | 0.01      | [100, 100, 50]      | [100, 100]           | 0.01 | 0.99        | 1.0      | 1e-08  | False      | 5000          | ([-5.0], [5.0])     | out/CartpoleTrial_CartpoleStabShort-v0 | 

Execute this command to obtain results:

    python main.py ddpg sim train parameters.csv out
    

`out` specifies the directory where the output result will be saved (this is strictly specified by the developer) for more information take a look at [config readme](src/config/Readme.md)

`train` the command to train the specified algorithm under given hyperparameters (the [parameters.csv](parameters.csv)) file

Your output should be something like this:

![Alt Text](https://i.imgur.com/fjlQHah.png)

And the given plot in your specified `outdir`:

![Imgur](https://i.imgur.com/abSj3MD.png)

To trial your models you can choose a model in the `outdir` that fits your needs.
The model names that you need are always called `actortarget` with some numbers that represent the obtained `reward` in a training session.

We choose the model that gained approx 10.000 reward and take the parameters.csv to a new folder called `test_model` and safe 
the policy as `policy` and take the specified `parameters.csv` into `test_model`.

Now we can execute the model and obtain our results graphically.

    python main.py ddpg sim trial test_model result 100
    


your reward plot for your policy will look like this:

![link](https://i.imgur.com/E4EMeeM.png)

You can see that the approximate `reward` is `10.000`

and the obtained policy looks like this (we only let the policy render once):

![gif](https://i.imgur.com/URL7zer.gif)

### real environment

If you want to execute this example on the real environment just train on the real environment
        
        python main.py ddpg sim train parameters.csv out
        
Put your `policy` and `parameters.csv` in a new directory for example `test_model`

        python main.py ddpg sim trial test_model result 1
        
After executing this command u will see similar results:

![gif](https://imgur.com/RY9QW4x.gif)
    

Have fun testing parameters and [writing your own algorithms](/src/config/Readme.md)

## Troubleshooting

If you have problems with training or trial sessions just make sure your output folders are empty and you always name the hyperparameters file as `parameters.csv` and the policy as `policy`.
