import numpy as np

class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4  # Number of rows in the grid
        self.cols = 12  # Number of columns in the grid
        self.start = (3, 0)  # Starting position
        self.goal = (3, 11)  # Goal position
        self.cliff = [(3, i) for i in range(1, 11)]  # Cliff area
        self.actions = ['U', 'D', 'L', 'R']  # Possible actions
        self.action_dict = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}  # Action effects

    def step(self, state, action):
        """Execute an action and return the next state, reward, and whether the episode is done."""
        row, col = state
        delta_row, delta_col = self.action_dict[action]
        next_state = (row + delta_row, col + delta_col)

        # Boundary check
        if next_state[0] < 0 or next_state[0] >= self.rows or next_state[1] < 0 or next_state[1] >= self.cols:
            next_state = state  # Stay in place if out of bounds

        # Check if the next state is in the cliff area
        if next_state in self.cliff:
            return self.start, -100, False  # Return to start with a penalty of -100
        elif next_state == self.goal:
            return next_state, 0, True  # Reach the goal with a reward of 0
        else:
            return next_state, -1, False  # Normal step with a reward of -1

    def reset(self):
        """Reset the environment to the starting state."""
        return self.start