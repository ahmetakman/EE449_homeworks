# This file will include the implementation of reinforcement learning homework to solve maze problem using temporal difference learning and q-learning

import numpy as np
import matplotlib.pyplot as plt

from utils import plot_value_function, plot_policy

class MazeEnvironment:
    def __init__(self):
        # Define the maze layout, rewards, action space (up, down, left, right)
        self.maze = np.array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                              [0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1],
                              [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
                              [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                              [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3],
                              [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1],
                              [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                              [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                              [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                              [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
        self.start_pos = (0, 0)  # Start position of the agent
        self.current_pos = self.start_pos
        self.state_penalty = -1
        self.trap_penalty = -100
        self.goal_reward = 100
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action):
        # Simulate the action based on stochastic environment
        if np.random.uniform(0, 1) <= 0.75:  # Probability of going for the chosen direction
            dx, dy = self.actions[action]
        elif np.random.uniform(0, 1) <= 0.05:  # Probability of going opposite of the chosen direction
            dx, dy = -self.actions[action]
        elif np.random.uniform(0, 1) <= 0.10:  # Probability of going each of perpendicular routes
            dx, dy = np.random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        else:
            dx, dy = self.actions[action]

        new_x, new_y = self.current_pos[0] + dx, self.current_pos[1] + dy

        # Check if the new position is valid
        if 0 <= new_x < self.maze.shape[0] and 0 <= new_y < self.maze.shape[1] and self.maze[new_x, new_y] != 1:
            self.current_pos = (new_x, new_y)

        # Determine reward based on new position
        if self.maze[self.current_pos] == 2:
            reward = self.trap_penalty
        elif self.maze[self.current_pos] == 3:
            reward = self.goal_reward
        else:
            reward = self.state_penalty

        return self.current_pos, reward

    

    # create an arbitrary value function
    def create_value_function(self):
        # Initialize the value function
        value_function = np.zeros(self.maze.shape)

        return value_function
    

# plot a value function
maze = MazeEnvironment()
value_function = maze.create_value_function()
plot_value_function(value_function, maze.maze)