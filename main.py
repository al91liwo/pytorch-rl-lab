import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG
from DDPG import batch_size_schedulers
import numpy as np
import numpy.random as rnd
torch.manual_seed(0)
env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)

dev = "cuda" if torch.cuda.is_available() else "cpu"


while True:
    ddpg = DDPG(env=env, episodes=200, warmup_samples=10000, buffer_size=50000, batch_size=50,
            actor_lr=1e-3, critic_lr=1e-2, actor_lr_decay=0.99, critic_lr_decay=0.99, noise_decay=0.95, epochs=1, batch_size_scheduler=1,
            actor_hidden_layers=[100, 200, 100], critic_hidden_layers=[100, 200, 100], device=dev)
    #ddpg = DDPG(env=env, episodes=150, warmup_samples=rnd.randint(10000,50000), buffer_size=50000+10000*rnd.randint(0,5), batch_size=32*rnd.randint(1,4),
    #        actor_lr=np.abs(rnd.normal(3e-4, 1e-4)), critic_lr=np.abs(rnd.normal(6e-4, 1e-4)), noise_decay=0.9+rnd.uniform(0.05,0.099), epochs=rnd.randint(1,3),
    #        actor_hidden_layers=[300, 300, 300], critic_hidden_layers=[300, 300, 300], batch_size_scheduler=rnd.randint(0, len(batch_size_schedulers)), device=dev)

    result_dirname = ddpg.train()
    ddpg.actor_target.eval()
    ddpg.load_model()
    episodes = 100
    rew = []
    with torch.no_grad():
        for step in range(episodes):
            done = False
            obs = env.reset()[0]
            total_reward = 0
            print(step)
            while not done:
                obs = ddpg.transformObservation(obs)
                state = torch.tensor(obs, dtype=torch.float32).to(dev)
            
                action = ddpg.actor_target(state).cpu().detach().numpy()
                obs, reward, done, _ = env.step(action)
                total_reward += reward

#                if step >= episodes - 1:
#                   env.render()


            rew.append(total_reward)
    env.close()


    print(sum(rew)/len(rew))
    ddpg.save_model(result_dirname, env_name)
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.plot(range(episodes), rew)
    plt.savefig(result_dirname+'/over100episodes.jpg')
    plt.clf()
    #plt.show()
