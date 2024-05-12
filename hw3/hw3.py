# This file will include the implementation of reinforcement learning homework to solve maze problem using temporal difference learning and q-learning

import numpy as np
import matplotlib.pyplot as plt
import random
from utils import plot_value_function, plot_policy

from tqdm import tqdm

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
            dx, dy = tuple([-1*x for x in self.actions[action]])
        elif np.random.uniform(0, 1) <= 0.10:  # Probability of going each of perpendicular routes
            dx, dy = self.actions[np.random.choice(list(self.actions.keys()))]
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



class MazeTD0(MazeEnvironment): # Inherited from MazeEnvironment
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha # Learning Rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration Rate
        self.episodes = episodes
        self.utility = {} # Utility values for states
        self.valid_actions = list(maze.actions.keys())
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Explore: Randomly choose an action
            return random.choice(self.valid_actions)
        else:
            # Exploit: Choose the best action based on current utility values
            return max(self.valid_actions, key=lambda a: self.utility.get((state, a), 0))

    def update_utility_value(self, current_state, reward, new_state):
        current_value = self.utility.get((current_state, None), 0)
        new_value = max(self.utility.get((new_state, a), 0) for a in self.valid_actions)
        td_target = reward + self.gamma * new_value
        self.utility[(current_state, None)] = current_value + self.alpha * (td_target - current_value)

    def run_episodes(self):
        for _ in tqdm(range(self.episodes)):
            current_state = self.maze.current_pos
            while True:
                action = self.choose_action(current_state)
                new_state, reward  = self.maze.step(action)
                self.update_utility_value(current_state, reward, new_state)
                current_state = new_state
                print(current_state)
                if self.maze.maze[current_state] == 3:
                    break
        return self.utility

# Create an instance of the Maze with TD(0) and run multiple episodes
maze = MazeEnvironment()
maze_td0 = MazeTD0(maze, alpha=0.1, gamma=0.95, epsilon=0.5, episodes=10000)
final_values = maze_td0.run_episodes()
