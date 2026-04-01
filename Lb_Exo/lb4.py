# DDPG implementation using Stable-Baselines3

import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Create environment
env = gym.make("Pendulum-v1")

# Add action noise (important for DDPG exploration)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1 * np.ones(n_actions))

# Create DDPG model
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train model
model.learn(total_timesteps=10000)

# Save model
model.save("ddpg_pendulum")

# Test the trained model
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
