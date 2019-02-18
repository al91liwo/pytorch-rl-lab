import gym
import torch
import matplotlib.pyplot as plt
import quanser_robots
from DDPG import DDPG
import ActorNetwork
import CriticNetwork

def load_model(actor_path, critic_path, actor_hidden_layers=[400, 300], critic_hidden_layers=[400, 300], action_dim, state_dim,  env_low, env_high):
    print('Loading models from {} and {}'.format(actor_path, critic_path))
    if actor_path is not None:
        actor = ActorNetwork([state_dim, *actor_hidden_layers, action_dim], torch.tensor(env_low), torch.tensor(env_high))
        actor.load_state_dict(torch.load(actor_path))
        actor.eval()
    if critic_path is not None:
        critic =CriticNetwork([state_dim + action_dim, *critic_hidden_layers, 1])
        critic.load_state_dict(torch.load(actor_path))
        critic.eval()
    return actor, critic


env_name = "CartpoleStabShort-v0"
env = gym.make(env_name)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
env_low = env.action_space.low
env_high = env.action_space.high


#TODO: define params
actor, critic = load_model("models/ddpg_actor_CartpoleStabShort-v0_2019-01-30 12:32:34.111933", None,
                           action_dim=action_dim, state_dim=state_dim, env_low=env_low, env_high=env_high)

episodes = 100
rew = []

for step in range(episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        state = torch.tensor(obs, dtype=torch.float32)

        action = actor(state).detach().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if step == episodes - 1:
            env.render()

    rew.append(total_reward)
env.close()

plt.plot(range(episodes), rew)
plt.show()
print(sum(rew) / len(rew))

