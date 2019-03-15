# Hyperparameters
|parameter              |default                |description                                                                                                                                          |
|-----------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
|run_id                 | 0                     | used in the output directory name                                                                                                                   |
|env                    |                       | the name of the gym environment to run the algorithm in                                                                                             |
|dirname                |out                    | the root directory for all outputs of the algorithm                                                                                                 |
|model                  |nn                     | possible values {nn, pnn, perfect}. nn: neural network, pnn: probabilistic neural network, perfect: an analytic implementation of the model. A perfect model needs to be supported by the algorithm for each individual environment. Currently only Pendulum and quanser cartpole are supported.|
|predict_horizon        | 20                    | the number of futuer actions the controler should optimize on                                                                                       |
|warmup_trials          | 1                     | number of rollouts with random actions performed to fill up the memory. Needs to be at least 1                                                      |
|learning_trials        | 20                    | number of trials to run the algorithm before terminating and saving the results                                                                     |
|cem_samples            | 400                   | the number of samples the CEM optimizer takes during one step                                                                                       |
|nelite                 | 40                    | the number of elite samples the CEM optimizer uses at each step                                                                                     |
|render                 | 0                     | if 0: does not render at all, if n: render every nth trial                                                                                          |
|max_memory             | 1000000               | maximum samples to keep in memory buffer                                                                                                            |
|device                 |cpu                    | the device to run all the pytorch operations on                                                                                                     |
|layers                 | [10]                  | the configuration of hidden layers. Only relevant if a neural network is selected as model.                                                         |
|batch_norm             | True                  | whether to batch normalize outputs of hidden layers. Only relevant if a neural network is                                                           |
|predicts_delta         | True                  |whether the model should learn the delta to the next state and then add it to the input. If False, it learns to output the next state directly       |
|propagate_probabilistic| False                 | if True and a probabilistic environment model is selected, the model will output a sample of its output distribution instead  of the mean           |
|variance_bound         | [1.e-5, 0.5]          | restricts the variance of a pnn                                                                                                                     |
|trial_horizon          | 1000                  | maximum length of a trial                                                                                                                           |
|weight_decay           | []                    | an array with same length as layers that specifies the decay appllied to the weights to each layer during updates                                   |
|lr                     | 1e-2                  | learnrate of an approximate model.                                                                                                                  |
|lr_min                 | 1e-5                  | minimum learnrate - when to stop learn rate decay                                                                                                   |
|lr_decay               | 1.                    | decay factor                                                                                                                                        |
|batch_size             | 50                    | the batchsized used during model training                                                                                                           |
|epochs                 | 1                     | the number of batches to train the network with during training phase of the algorithm                                                              |
|logging                | False                 | whether to log training information                                                                                                                 |
|plotting               | False                 | !ONLY FOR DIRTY DEBUGGING whether to plot graph that diagnose model training progress                                                                                         |
