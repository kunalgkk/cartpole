import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Create environment
env = gym.make("Pendulum-v1", render_mode="rgb_array")  # render_mode for visualization

# Define Actor and Critic networks
actor = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Tanh()
)
critic = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 1)
)

opt_a = optim.Adam(actor.parameters(), lr=0.001)
opt_c = optim.Adam(critic.parameters(), lr=0.001)

episodes = 20
max_steps = 200
gamma = 0.9

reward_history = []

for ep in range(episodes):
    s, _ = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        s_t = torch.tensor(s, dtype=torch.float32)
        a = actor(s_t).detach().numpy()

        ns, r, t, tr, _ = env.step(a)
        done = t or tr

        sa = torch.tensor(np.append(s, a), dtype=torch.float32)
        # nsa = torch.tensor(np.append(ns, a), dtype=torch.float32)
        ns_t = torch.tensor(ns, dtype=torch.float32)
        next_a = actor(ns_t).detach()
        nsa = torch.cat([ns_t, next_a], dim=0)

        # Critic update
        target = r + gamma * critic(nsa)
        loss_c = (critic(sa) - target.detach())**2
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

        # Actor update
        #loss_a = -critic(torch.tensor(np.append(s, actor(s_t)), dtype=torch.float32))
        a_pred = actor(s_t)                      # tensor with grad
        sa_actor = torch.cat([s_t, a_pred], dim=0)
        loss_a = -critic(sa_actor)
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()

        s = ns
        total_reward += r
        if done: break

    reward_history.append(total_reward)
    print(f"Episode {ep} Reward: {total_reward:.2f}")

env.close()

# Plot rewards over episodes
plt.figure(figsize=(10,5))
plt.plot(range(episodes), reward_history, marker='o')
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
