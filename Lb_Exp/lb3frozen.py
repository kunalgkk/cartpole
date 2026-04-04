import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True)

state_size = env.observation_space.n
action_size = env.action_space.n

# Convert state to one-hot vector
def one_hot(state):
    vec = np.zeros(state_size)
    vec[state] = 1
    return vec

# Neural Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
episodes = 2000
batch_size = 64

memory = deque(maxlen=10000)

# Training
for episode in range(episodes):
    state, _ = env.reset()
    state = one_hot(state)
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_vec = one_hot(next_state)

        memory.append((state, action, reward, next_state_vec, done))
        state = next_state_vec

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

            q_values = model(states)
            next_q_values = model(next_states)

            target_q = q_values.clone()

            for i in range(batch_size):
                if dones[i]:
                    target_q[i, actions[i]] = rewards[i]
                else:
                    target_q[i, actions[i]] = rewards[i] + gamma * torch.max(next_q_values[i])

            loss = loss_fn(q_values, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training completed using DQN")