import numpy as np

# SARSA Agent
class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.env = env  # Environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))  # Initialize Q-table

    def choose_action(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)  # Explore: choose a random action
        else:
            return self.env.actions[np.argmax(self.q_table[state])]  # Exploit: choose the best action

    def learn(self, num_episodes):
        """Train the agent using the SARSA algorithm."""
        rewards = []  # Store rewards for each episode
        all_paths = []  # Store paths for each episode

        for episode in range(num_episodes):
            state = self.env.reset()  # Reset the environment
            action = self.choose_action(state)  # Choose an initial action
            total_reward = 0
            path = [state]  # Store the path taken in this episode

            while True:
                next_state, reward, done = self.env.step(state, action)  # Take a step
                next_action = self.choose_action(next_state)  # Choose the next action
                total_reward += reward
                path.append(next_state)

                # SARSA update
                current_action_index = self.env.actions.index(action)
                next_action_index = self.env.actions.index(next_action)
                self.q_table[state][current_action_index] += self.alpha * (reward + self.gamma * self.q_table[next_state][next_action_index] - self.q_table[state][current_action_index])

                state, action = next_state, next_action  # Move to the next state and action
                if done:
                    break

            rewards.append(total_reward)
            all_paths.append(path)  # Save the path for visualization

        return rewards, all_paths


# Q-learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=1.0, epsilon=0.1):
        self.env = env  # Environment
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.rows, env.cols, len(env.actions)))  # Initialize Q-table

    def choose_action(self, state):
        """Epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)  # Explore: choose a random action
        else:
            return self.env.actions[np.argmax(self.q_table[state])]  # Exploit: choose the best action

    def learn(self, num_episodes):
        """Train the agent using the Q-learning algorithm."""
        rewards = []  # Store rewards for each episode
        all_paths = []  # Store paths for each episode

        for episode in range(num_episodes):
            state = self.env.reset()  # Reset the environment
            total_reward = 0
            path = [state]  # Store the path taken in this episode

            while True:
                action = self.choose_action(state)  # Choose an action
                next_state, reward, done = self.env.step(state, action)  # Take a step
                total_reward += reward
                path.append(next_state)

                # Q-learning update
                current_action_index = self.env.actions.index(action)
                self.q_table[state][current_action_index] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][current_action_index])

                state = next_state  # Move to the next state
                if done:
                    break

            rewards.append(total_reward)
            all_paths.append(path)  # Save the path for visualization

        return rewards, all_paths
