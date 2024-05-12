import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap

def plot_value_function(value_function, maze):
    mask = np.zeros_like(value_function, dtype=bool)
    mask[maze == 1] = True  # Mask obstacles
    mask[maze == 2] = True  # Mask the trap
    mask[maze == 3] = True  # Mask the goal

    trap_position = tuple(np.array(np.where(maze == 2)).transpose(1,0))
    goal_position = np.where(maze == 3) 
    obs_position = tuple(np.array(np.where(maze == 1)).transpose(1,0))

    plt.figure(figsize=(10, 10))
    #cmap = sns.light_palette("green", as_cmap=True)
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    ax = sns.heatmap(value_function, mask=mask, annot=True, fmt=".1f", cmap=cmap,
                     cbar=False, linewidths=1, linecolor='black')
    ax.add_patch(plt.Rectangle(goal_position[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for t in trap_position:
        ax.add_patch(plt.Rectangle(t[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for o in obs_position:
        ax.add_patch(plt.Rectangle(o[::-1], 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    ax.set_title("Value Function")
    plt.show()

def plot_policy(value_function, maze):
    policy_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    policy_grid = np.full(maze.shape, '', dtype='<U2')
    actions = ['up', 'down', 'left', 'right']

    trap_position = tuple(np.array(np.where(maze == 2)).transpose(1,0))
    goal_position = np.where(maze == 3) 
    obs_position = tuple(np.array(np.where(maze == 1)).transpose(1,0))

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1 or (i, j) == goal_position:
                continue  # Skip obstacles and the goal
            best_action = None
            best_value = float('-inf')
            for action in actions:
                next_i, next_j = i, j
                if action == 'up':
                    next_i -= 1
                elif action == 'down':
                    next_i += 1
                elif action == 'left':
                    next_j -= 1
                elif action == 'right':
                    next_j += 1
                if 0 <= next_i < maze.shape[0] and 0 <= next_j < maze.shape[1]:
                    if value_function[next_i][next_j] > best_value:
                        best_value = value_function[next_i][next_j]
                        best_action = action
            if best_action:
                policy_grid[i][j] = policy_arrows[best_action]

    mask = np.zeros_like(value_function, dtype=bool)
    mask[maze == 1] = True  # Mask obstacles
    mask[maze == 2] = True  # Mask the trap
    mask[maze == 3] = True  # Mask the goal

    plt.figure(figsize=(10, 10))
    #cmap = sns.light_palette("green", as_cmap=True)
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    ax = sns.heatmap(value_function, mask=mask, annot=policy_grid, fmt="", cmap=cmap,
                     cbar=False, linewidths=1, linecolor='black')
    ax.add_patch(plt.Rectangle(goal_position[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkgreen'))
    for t in trap_position:
        ax.add_patch(plt.Rectangle(t[::-1], 1, 1, fill=True, edgecolor='black', facecolor='darkred'))
    for o in obs_position:
        ax.add_patch(plt.Rectangle(o[::-1], 1, 1, fill=True, edgecolor='black', facecolor='gray'))
    ax.set_title("Policy Map")
    plt.show()

