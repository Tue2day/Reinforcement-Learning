import numpy as np
import matplotlib.pyplot as plt

class MazeSolver:
    def __init__(self):
        # Modeling maze
        self.maze = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
        self.start = (2, 0)
        self.end = (6, 7)

        # Parameter settings
        self.gamma = 0.9  # Discount factor
        self.reward = -1  # Reward for each step
        self.threshold = 0.0001  # Convergence threshold

        # Action definitions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Value matrix V
        self.V = np.full((len(self.maze), len(self.maze[0])), -np.inf)
        self.initialize()

    def initialize(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == 1:
                    self.V[i][j] = 0

    def is_valid(self, state):
        x, y = state
        return 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 1

    def get_next_state(self, state, action):
        return (state[0] + action[0], state[1] + action[1])

    def value_iteration(self):
        while True:
            delta = 0
            for i in range(len(self.maze)):
                for j in range(len(self.maze[0])):
                    if (i, j) == self.end or self.maze[i][j] == 0:
                        continue
                    # Calculate the maximum value for the current state
                    max_value = float('-inf')
                    for action in self.actions:
                        next_state = self.get_next_state((i, j), action)
                        if self.is_valid(next_state):
                            max_value = max(max_value, self.reward + self.gamma * self.V[next_state])

                    # Update the value of the current state
                    delta = max(delta, abs(self.V[i][j] - max_value))
                    self.V[i][j] = max_value
            if delta < self.threshold:
                break

    def find_path(self):
        path = [self.start]
        state = self.start
        while state != self.end:
            max_value = float('-inf')
            next_state = None
            for action in self.actions:
                new_state = self.get_next_state(state, action)
                if self.is_valid(new_state) and self.reward + self.gamma * self.V[new_state] > max_value:
                    max_value = self.reward + self.gamma * self.V[new_state]
                    next_state = new_state
            if next_state is None:
                print("No valid path found!")
                return []
            path.append(next_state)
            state = next_state
        return path

    def visualize_path(self, path):
        # Visualize the maze and the path
        fig, ax = plt.subplots()

        # Draw the maze
        maze_array = np.array(self.maze)
        ax.imshow(maze_array, cmap='binary', vmin=0, vmax=1, interpolation='nearest')

        # Mark the start and end points
        ax.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')  # Start point in green
        ax.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')  # End point in red

        # Draw the path
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')  # Path in blue

        # Set grid lines
        ax.set_xticks(np.arange(-0.5, len(self.maze[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.maze), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        # Show legend
        ax.legend()

        # Display the plot
        plt.title("Maze with Path")
        plt.show()

if __name__ == '__main__':
    # Create a MazeSolver object
    solver = MazeSolver()

    # Run value iteration
    solver.value_iteration()

    # Find and print the path
    path = solver.find_path()
    print("Found path:", path)

    # Visualize the path
    solver.visualize_path(path)
