# Example
Obtain some examples of your algorithms.

### DDPG
The algorithm is explained right here -> [DDPG](../../algorithm/DDPG/Readme.md).

#### trial
To start a trial session you can for example use this command.

        python main.py ddpg sim trial src/config/example/trial src/config/example/trial_out 1

The command will save a reward plot in [this directory](/src/config/example) in the folder [out](/src/config/example/out)
as you may test by yourself.

Your specified folder needs always to specify a `parameters.csv` and a `policy` (the name is mandatory).
Where policy defines a model (neural network weights) for the policy and parameters are the used parameters
 to load the model and execute either in training or in trial mode.

This example was executed in `trial` mode as you can read in the command.

#### test
To start a training session you can for example use this command.
   
        python main.py ddpg sim train src/config/example/train/parameters.csv src/config/example/train_out

After that you can obtain a `actortarget` model that fits your needs and take the `parameters.csv` to a new folder
and execute it as in the trial section.

### MPC
The algorithm is explained right here -> [MPC](../../algorithm/MPC/Readme.md)

#### test
To start a training session you can use this command.
   
        python main.py mpc sim train src/config/example/train/testmpc.csv src/config/example/train_out

After that `src/config/example/train_out/textmpc_CartpoleStabShort-v0` will contain the results of the training. This is a file `rewarddata` containing the total episode rewards and a file `trajectories` containing all trial trajectories, each on one line, in order.