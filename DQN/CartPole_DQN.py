import gymnasium as gym
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from torch.xpu import device


class QNet(nn.Module):
    """Q-network for DQN."""
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer.clear()


class DQNAgent:
    """DQN agent."""
    def __init__(self, state_size, action_size, hidden_size, device, args):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.args = args
        self.learn_step = 0

        # Networks
        self.eval_net = QNet(state_size, hidden_size, action_size).to(device)
        self.target_net = QNet(state_size, hidden_size, action_size).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(args.capacity)

        # Epsilon for epsilon-greedy
        self.eps = args.eps

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() <= self.eps:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.eval_net(state)
                return torch.argmax(q_values).item()

    def store_transition(self, *transition):
        """Store a transition in the replay buffer."""
        self.buffer.push(*transition)

    def update_target_net(self):
        """Update the target network."""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        """Train the agent using a batch of experiences."""
        # Sample a batch of transitions
        batch = self.buffer.sample(self.args.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute q_eval with eval_net and q_next with target_net
        q_eval = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(next_states).detach().max(1)[0]
        q_target = rewards + self.args.gamma * (1 - dones) * q_next

        # Compute loss and update the network
        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.eps = max(self.args.eps_min, self.eps * self.args.eps_decay)

        # Update target network
        if self.learn_step % self.args.update_target == 0:
            self.update_target_net()
        self.learn_step += 1


def train(env, agent, args):
    """Train the DQN agent."""
    scores = []
    for episode in range(args.n_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        step_cnt = 0

        while not done and step_cnt < 500:
            step_cnt += 1
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            # Train the agent if enough data is available
            if len(agent.buffer) >= args.batch_size:
                agent.learn()

        scores.append(episode_reward)
        print(f"Episode: {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.eps:.4f}")

    return scores


def plot_results(scores, args):
    """Plot the training results."""
    plt.plot(range(args.n_episodes), scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DQN Performance on {args.env}")
    plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="DQN for CartPole-v1")
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--hidden", default=64, type=int, help="Hidden layer size")
    parser.add_argument("--n_episodes", default=500, type=int, help="Number of episodes")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--capacity", default=5000, type=int, help="Replay buffer capacity")
    parser.add_argument("--eps", default=1.0, type=float, help="Initial epsilon for epsilon-greedy")
    parser.add_argument("--eps_min", default=0.05, type=float, help="Minimum epsilon")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--eps_decay", default=0.999, type=float, help="Epsilon decay rate")
    parser.add_argument("--update_target", default=100, type=int, help="Target network update frequency")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment and agent
    env = gym.make(args.env, render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, args.hidden, device, args)

    # Train the agent
    scores = train(env, agent, args)

    # Plot results
    plot_results(scores, args)

    env.close()


if __name__ == "__main__":
    main()