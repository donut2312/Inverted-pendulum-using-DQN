import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (np.array(states), actions, rewards, np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy policy
def select_action(state, epsilon, q_net, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state)
            return q_values.argmax().item()

# Training loop
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    rewards = []

    for episode in range(EPISODES):
        state = env.reset()[0]
        total_reward = 0

        while True:
            action = select_action(state, epsilon, q_net, action_dim)
            next_state, reward, done, _, _ = env.step(action)

            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            # Train
            if len(buffer) > BATCH_SIZE:
                states, actions, rewards_batch, next_states, dones = buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_batch = torch.FloatTensor(rewards_batch).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Current Q-values
                current_q = q_net(states).gather(1, actions)

                # Target Q-values
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards_batch + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()
    return rewards

# Run training
rewards = train()

# Plot reward trend
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN on Inverted Pendulum (CartPole)")
plt.show()
