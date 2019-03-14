# Example
Obtain some examples of your algorithms.

### DDPG
The algorithm is explained right here -> [DDPG](../../algorithm/DDPG/Readme.md).

#### trial
To start a trial session you can for example use this command.

        python main.py ddpg sim trial src/config/example/trial src/config/example/trial_out 100

The command will save a reward plot in [this directory](/src/config/example) in the folder [out](/src/config/example/out)
as you may test by yourself.

Your specified folder needs always to specify a `parameters.csv` and a `policy` (the name is mandatory).
Where policy defines a model (neural network weights) for the policy and parameters are the used parameters
 to load the model and execute either in training or in trial mode.

This example was executed in `trial` mode as you can read in the command.

#### test
To start a training session you can for example use this command.
   
        python main.py ddpg sim train src/config/example/train src/config/example/train_out

The plot will look like something like this:


Insert some gifs or plots with hyperparameters 

### MPC
The algorithm is explained right here -> [MPC](../../algorithm/MPC/Readme.md)

Insert some gifs or plots with hyperparameters. 