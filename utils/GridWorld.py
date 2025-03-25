import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

STEP_REWARD = -1
END_REWARD = 10
CMAP1 = ListedColormap(['white', 'yellow', 'green'])
CMAP2 = ListedColormap(['white', 'yellow', 'green', 'orange', 'red', 'purple'])
CMAP3 = ListedColormap(['white', 'yellow', 'green', 'red'])

# 1 is start, 2 is end, 3 is permanet obstacle, 4 is removable obstacle, 5 
# is lever to remove obstacle

START_STATES = [(0, 0), (0, 0), (0, 0), (0, 0)]
END_STATES = [(2, 2), (3, 3), (3, 3), (2, 2)]

WORLD0 = np.array([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 2]], dtype=int)

WORLD1 = np.array([[1, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 2]], dtype=int)

WORLD2 = np.array([[1, 0, 0, 0],
                   [0, 3, 3, 0],
                   [0, 0, 0, 3],
                   [0, 3, 0, 2],
                   [0, 0 ,0 ,0]], dtype=int)

WORLD3 = np.array([[1, 0, 5],
                   [0, 3, 3],
                   [0, 4, 2]], dtype=int)

class GridWorld:
    def __init__(self, grid, start, end):
        self.world = grid.copy()
        self.grid = grid.copy()
        self.start = start
        self.end = end
        self.state = start
        self.prev_state = None
        self.lever = False

    def get_reward(self):
        if self.state == self.end:
            return END_REWARD
        if self.prev_state is None:
            self.prev_state = self.state
            return STEP_REWARD * 4
        if self.prev_state == self.state:
            return -5
        self.prev_state = self.state
        if self.lever:
            self.lever = False
            return STEP_REWARD
        return STEP_REWARD
    
    def reset(self):
        self.state = self.start
        self.grid = self.world.copy()
        self.prev_state = None
        self.lever = False

    def remove_obstacles(self):
        self.lever = True
        self.grid[self.grid == 4] = 0

    def update_state(self, action):
        i, j = self.state
        max_i, max_j = self.grid.shape
        if action == 'UP':
            i = max(0, i - 1)
        if action == 'DOWN':
            i = min(max_i - 1, i + 1)
        if action == 'LEFT':
            j = max(0, j - 1)
        if action == 'RIGHT':
            j = min(max_j - 1, j + 1)
        if self.grid[i, j] == 3 or self.grid[i, j] == 4:    # Obstacle
                return
        if self.grid[i, j] == 5:
            self.grid[i, j] = 0
            self.remove_obstacles()
        self.state = i, j

    def get_cell(self):
        return self.grid[self.state]

    def visualize(self, fname):
        # Plot the grid
        fig, ax = plt.subplots()
        if 5 in self.grid:
            ax.matshow(self.grid, cmap=CMAP2)
        elif 3 in self.grid:
            ax.matshow(self.grid, cmap=CMAP3)
        else:
            ax.matshow(self.grid, cmap=CMAP1)
        
        # Set gridlines to match cell boundaries
        #ax.set_xticks(np.arange(self.grid.shape[1] + 1) - 0.5, minor=True)
        #ax.set_yticks(np.arange(self.grid.shape[0] + 1) - 0.5, minor=True)
        #ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

        plt.savefig(fname)
        plt.close()

    def create_path(self, agent):
        pathGrid = self.grid.copy()
        max_steps = self.grid.shape[0] * self.grid.shape[1] * 10
        state = self.start
        steps = 0
        while state != self.end:
            action = agent.get_action(state, 0)
            print(action)
            self.update_state(action)
            state = self.state
            steps += 1
            if steps > max_steps:
                break
        return pathGrid

def get_world(world):
    return globals()["WORLD" + str(world)], START_STATES[world], END_STATES[world]

def get_random_world():
    n_rows, n_cols = np.random.randint(3, 200, 2)
    grid = np.zeros((n_rows, n_cols), dtype=int)
    start = (np.random.randint(n_rows), np.random.randint(n_cols))
    end = (np.random.randint(n_rows), np.random.randint(n_cols))
    while start == end:
        end = (np.random.randint(n_rows), np.random.randint(n_cols))
    n_obstacles = np.random.randint(1, n_rows * n_cols // 2)
    for _ in range(n_obstacles):
        i, j = np.random.randint(n_rows), np.random.randint(n_cols)
        while grid[i, j] != 0:
            i, j = np.random.randint(n_rows), np.random.randint(n_cols)
        grid[i, j] = 3
    grid[start] = 1
    grid[end] = 2
    return grid, start, end
