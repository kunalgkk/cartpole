import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

episodes = 300
batch_size = 32
scores = []

for e in range(episodes):
    state = env.reset()[0]
    total_reward = 0

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    scores.append(total_reward)
    print(f"Episode {e+1}/{episodes}, Score: {total_reward}")

plt.plot(scores)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("DQN Training Performance")
plt.show()
