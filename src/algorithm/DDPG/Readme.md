# DDPG

This implementation follows the [original paper](https://arxiv.org/abs/1509.02971) with some new hyperparameters as [described here](/src/config/DDPG/Readme.md).

You can start a new instance of the DDPG algorithm like this:

    # initialize a DDPG instance
    ddpg = DDPG(env="Pendulum-v0", action_space_limits=([-10.], [10.]), is_quanser_env=False)
    
    # start training
    ddpg.train()
    
You can now use the trained model of this instance on a environment:

    env = gym.make("Pendulum-v0")
    
    # get out of training mode into evaluation mode
    ddpg.actor_network.eval()
    
    obs = env.reset()
    done = False
    while not done:
        state = obs
        action = ddpg.forward_actor_network(ddpg.actor_network, state)
        obs, reward, done, _ = env.step(action)
        
        env.render()

elf, layers, actionspace_low, actionspace_high, activations=None, final_w=0.003,
                 batch_norm=True
                 
The [actor](ActorNetwork.py) and [critic](CriticNetwork.py) networks are neural networks that can be fed with numerical lists
that represent their layer structure.

    actor_network = ActorNetwork(layers=[3, 10, 10, 1], actionspace_low=-10, actionspace_high=10)
    
These networks are for example used to create a neural network with a input layer of 3 nodes and a output node of 1.
The actor_network creates all layers with [ReLU6](https://pytorch.org/docs/stable/nn.html), the last layer will be clamped with a tanh layer
that will use actionspace_low and actionspace_high as min and max values.

The CriticNetwork is similar to the ActorNetwork. 