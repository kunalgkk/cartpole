# Asynchronous Advantage Actor-Critic (A3C) Example
# Environment: CartPole-v1
# Python + PyTorch + Gymnasium

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-4
UPDATE_GLOBAL_ITER = 5
MAX_EP = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_ep, res_queue, wid):
        super(Worker, self).__init__()
        self.name = f"Worker-{wid}"
        self.global_net = global_net
        self.optimizer = optimizer
        self.local_net = ActorCritic(state_dim, action_dim).to(device)
        self.global_ep = global_ep
        self.res_queue = res_queue
        self.env = gym.make(ENV_NAME)

    def run(self):
        total_step = 1
        while self.global_ep.value < MAX_EP:
            state, _ = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_reward = 0

            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                probs, _ = self.local_net(state_tensor)
                action = torch.multinomial(probs, 1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_reward += reward
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update_global(next_state, done, buffer_s, buffer_a, buffer_r)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    self.local_net.load_state_dict(self.global_net.state_dict())

                state = next_state
                total_step += 1

                if done:
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.res_queue.put(ep_reward)
                    break

        self.res_queue.put(None)

    def update_global(self, next_state, done, buffer_s, buffer_a, buffer_r):
        if done:
            v_s_ = 0
        else:
            s = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            _, v = self.local_net(s)
            v_s_ = v.item()

        targets = []
        for r in reversed(buffer_r):
            v_s_ = r + GAMMA * v_s_
            targets.append(v_s_)
        targets.reverse()

        states = torch.FloatTensor(buffer_s).to(device)
        actions = torch.LongTensor(buffer_a).to(device)
        targets = torch.FloatTensor(targets).to(device)

        probs, values = self.local_net(states)
        values = values.squeeze()

        td = targets - values
        critic_loss = td.pow(2)

        log_probs = torch.log(probs.gather(1, actions.view(-1, 1)).squeeze())
        actor_loss = -log_probs * td.detach()

        loss = (critic_loss + actor_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    global_net = ActorCritic(state_dim, action_dim).to(device)
    global_net.share_memory()
    optimizer = optim.Adam(global_net.parameters(), lr=LR)

    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()

    workers = []
    for i in range(mp.cpu_count()):
        w = Worker(global_net, optimizer, global_ep, res_queue, i)
        w.start()
        workers.append(w)

    rewards = []
    finished_workers = 0

    while finished_workers < len(workers):
        r = res_queue.get()
        if r is None:
            finished_workers += 1
        else:
            rewards.append(r)
            print(f"Episode Reward: {r}")

    for w in workers:
        w.join()