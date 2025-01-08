from environment import CliffWalkingEnv
from agent import SarsaAgent, QLearningAgent
from visualization import visualize_path, plot_rewards

if __name__ == '__main__':
    # Initialize the environment
    env = CliffWalkingEnv()

    # Initialize the agents
    sarsa_agent = SarsaAgent(env, alpha=0.1, gamma=1.0, epsilon=0.1)
    q_learning_agent = QLearningAgent(env, alpha=0.1, gamma=1.0, epsilon=0.1)

    # Train the agents
    num_episodes = 500
    rewards_sarsa, paths_sarsa = sarsa_agent.learn(num_episodes)
    rewards_q_learning, paths_q_learning = q_learning_agent.learn(num_episodes)

    # Visualize the final paths
    visualize_path(env, paths_sarsa[-1], "SARSA Final Path")
    visualize_path(env, paths_q_learning[-1], "Q-Learning Final Path")

    # Plot the rewards
    plot_rewards(rewards_sarsa, rewards_q_learning)