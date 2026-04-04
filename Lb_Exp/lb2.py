import gymnasium as gym
import numpy as np
import random

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize Q-table and eligibility traces
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))
E = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 0.1      # exploration rate
lambda_ = 0.8      # eligibility trace decay
episodes = 5000

for episode in range(episodes):
    state = env.reset()[0]

    # Choose action using epsilon-greedy
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    # Reset eligibility traces
    E *= 0
    done = False
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state])

        td_error = reward + (0 if done else gamma * Q[next_state, next_action]) - Q[state, action]

        E[state, action] += 1
        Q += alpha * td_error * E
        E *= gamma * lambda_

        state = next_state
        action = next_action


print("Training completed using SARSA-Lambda")
print("Learned Q-table:")
print(Q)