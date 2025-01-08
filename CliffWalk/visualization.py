import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_path(env, path, title):
    """Visualize the path taken by the agent."""
    fig, ax = plt.subplots()
    cmap = ListedColormap(['white', 'black', 'red', 'green'])  # Define color map

    # Create the environment matrix
    env_matrix = np.zeros((env.rows, env.cols))
    for (row, col) in env.cliff:
        env_matrix[row][col] = 1  # Mark cliff areas
    env_matrix[env.goal] = 2  # Mark the goal
    env_matrix[env.start] = 3  # Mark the start

    # Plot the environment
    ax.imshow(env_matrix, cmap=cmap, interpolation='nearest')

    # Plot the path
    for (row, col) in path:
        ax.plot(col, row, 'bo', markersize=10)  # Mark path points with blue dots
        plt.pause(0.1)  # Pause to visualize step-by-step

    # Mark the start and goal
    ax.plot(env.start[1], env.start[0], 'go', markersize=15, label='Start')  # Start in green
    ax.plot(env.goal[1], env.goal[0], 'ro', markersize=15, label='Goal')  # Goal in red

    # Set grid lines
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Add legend
    ax.legend()

    # Set title
    plt.title(title)
    plt.show()


def plot_rewards(rewards_sarsa, rewards_q_learning):
    """Plot the rewards over episodes for SARSA and Q-learning."""
    plt.plot(range(len(rewards_sarsa)), rewards_sarsa, label='SARSA')
    plt.plot(range(len(rewards_q_learning)), rewards_q_learning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.show()
