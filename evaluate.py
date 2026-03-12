import gymnasium as gym
import torch
from dqn_agent import DQNAgent

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

# Load trained model
agent.model.load_state_dict(torch.load("dqn_cartpole.pth"))
agent.model.eval()

episodes = 20
total_rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0

    for t in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    total_rewards.append(total_reward)

print("Average Reward After Training:", sum(total_rewards)/episodes)
