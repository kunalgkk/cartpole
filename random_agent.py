import gymnasium as gym

env = gym.make("CartPole-v1")

episodes = 50
total_rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0

    for t in range(500):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    total_rewards.append(total_reward)

print("Average Reward (Random Policy):", sum(total_rewards)/episodes)
