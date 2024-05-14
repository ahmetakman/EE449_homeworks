# This file will include the implementation of reinforcement learning homework to solve maze problem using temporal difference learning and q-learning

import numpy as np
import matplotlib.pyplot as plt
import random
from utils import plot_value_function, plot_policy

from tqdm import tqdm


class MazeEnvironment:
    def __init__(self):
        # Define the maze layout, rewards, action space (up, down, left, right)
        self.maze = np.array(
            [
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1],
                [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3],
                [0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
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
        random_val = np.random.uniform(0, 1)
        if random_val <= 0.75:  # Probability of going for the chosen direction
            dx, dy = self.actions[action]
        elif (
            0.75 < random_val <= 0.80
        ):  # Probability of going opposite of the chosen direction
            dx, dy = tuple([-1 * x for x in self.actions[action]])
        elif (
            0.80 < random_val <= 0.90
        ):  # Probability of going each of perpendicular routes
            dy, dx = self.actions[action]  # Swap the x and y coordinates
        else:
            dy, dx = self.actions[action]
            dx, dy = -dy, -dx

        new_x, new_y = self.current_pos[0] + dx, self.current_pos[1] + dy

        # Check if the new position is valid
        if (
            (0 <= new_x < self.maze.shape[0])
            and (0 <= new_y < self.maze.shape[1])
            and (self.maze[new_x, new_y] != 1)
        ):
            self.current_pos = (new_x, new_y)
        else:  # If the new position is invalid, stay in the current position
            self.current_pos = self.current_pos
            return self.current_pos, self.state_penalty

        # Determine reward based on new position
        if self.maze[self.current_pos] == 2:
            reward = self.trap_penalty
        elif self.maze[self.current_pos] == 3:
            reward = self.goal_reward
        else:
            reward = self.state_penalty

        return self.current_pos, reward


class MazeTD0(MazeEnvironment):  # Inherited from MazeEnvironment
    def __init__(self, maze, alpha=0.1, gamma=0.95, epsilon=0.2, episodes=10000):
        super().__init__()
        self.maze = maze
        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration Rate
        self.episodes = episodes
        self.utility =  5*np.ones((12,12)) #- 30 * np.pad(maze.maze, [(0, 1), (0, 1)], mode='constant') # Utility values for states
        # self.utility[np.where(maze.maze == 3)] = 10000
        self.utility[np.where(maze.maze == 1)] = -1000
        self.valid_actions = list(maze.actions.keys())

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Explore: Randomly choose an action
            return random.choice(self.valid_actions)
        else:
            # Exploit: Choose the best action based on current utility values
            return np.argmax(
                [
                    self.utility[state[0] + dx, state[1] + dy]
                    for dx, dy in self.maze.actions.values()
                ]
            )

    def update_utility_value(self, current_state, reward, new_state):
        current_value = self.utility[current_state[0], current_state[1]]
        new_value = self.utility[new_state[0], new_state[1]]
        td_target = reward + self.gamma * new_value
        self.utility[current_state[0], current_state[1]] = current_value + self.alpha * (td_target - current_value)
        

    def run_episodes(self):
        for _ in tqdm(range(self.episodes)):
            self.maze.reset()

            while True:
                current_state = self.maze.current_pos

                action = self.choose_action(current_state)
                new_state, reward = self.maze.step(action)
                self.update_utility_value(current_state, reward, new_state)
                current_state = new_state
                # print(current_state)
                if self.maze.maze[current_state] == 3 or self.maze.maze[current_state] == 2:
                    break
            if _ % 1000 == 0:
                plot_value_function(self.utility[0:11, 0:11], self.maze.maze)
        return self.utility


# Create an instance of the Maze with TD(0) and run multiple episodes
maze = MazeEnvironment()
maze_td0 = MazeTD0(maze, alpha=0.1, gamma=0.95, epsilon=0.3, episodes=10000)
final_values = maze_td0.run_episodes()
print(final_values)
final_values = final_values[0:11, 0:11]

plot_value_function(final_values, maze.maze)
plot_policy(final_values, maze.maze)
