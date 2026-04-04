import gymnasium as gym
import numpy as np
import random

states = 6
actions = 2
Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500

rewards = np.array([
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 1],
    [0, 0],
    [0, 1]
])

for episode in range(episodes):
    state = random.randint(0, states - 1)
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(Q[state])

        reward = rewards[state][action]
        next_state = (state + action) % states

        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state
        if reward == 1:
            done = True

print(Q)