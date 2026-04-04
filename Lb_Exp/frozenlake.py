import gymnasium as gym
import numpy as np
import random

env = gym.make("FrozenLake-v1", is_slippery=True)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8
gamma = 0.95
epsilon = 0.1
episodes = 5000

for episode in range(episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _, _ = env.step(action)

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Q-table learned")