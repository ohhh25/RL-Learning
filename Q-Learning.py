import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from utils.q_table import plot_q_table

# Constants
START = (0, 0)
END = (2, 2)
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
GRID = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

STEP_REWARD = -1
END_REWARD = 10
MAX_STEPS = len(GRID) * len(GRID[0]) * 10

# Hyperparameters
learning_rate = 0.1
episodes = 10000

class GridWorld:
    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.state = start

    def get_reward(self):
        if self.state == self.end:
            return END_REWARD
        return STEP_REWARD
    
    def reset(self):
        self.state = self.start

    def update_state(self, action):
        i, j = self.state
        max_i, max_j = len(self.grid), len(self.grid[0])
        if action == 'UP':
            self.state = max(0, i - 1), j
        if action == 'DOWN':
            self.state = min(max_i - 1, i + 1), j
        if action == 'LEFT':
            self.state = i, max(0, j - 1)
        if action == 'RIGHT':
            self.state = i, min(max_j - 1, j + 1)
    
class Agent:
    def __init__(self, grid, epsilon=1.0, epsilon_decay=0.9, min_epsilon=1e-3,
                 discount=0.9):
        self.q_table = np.zeros((len(grid), len(grid[0]), 4))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount = discount

    def get_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[np.argmax(self.q_table[state[0], state[1]])]
    
    def update_q_table(self, state, action, reward, next_state):
        i, j = state
        k = ACTIONS.index(action)
        next_i, next_j = next_state

        target = reward + self.discount * np.max(self.q_table[next_i, next_j])
        error = target - self.q_table[i, j, k]
        self.q_table[i, j, k] += learning_rate * error

def train(grid_world, agent):
    steps = []
    for episode in tqdm(range(episodes)):
        steps_taken = 0
        state = grid_world.state
        while grid_world.state != grid_world.end:
            action = agent.get_action(state, agent.epsilon)
            grid_world.update_state(action)
            next_state = grid_world.state
            reward = grid_world.get_reward()
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            steps_taken += 1
            if steps_taken > MAX_STEPS:
                break
        steps.append(steps_taken)
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.min_epsilon)
        grid_world.reset()
    return steps

def post_training(steps, QTable):
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps vs Episode')
    plt.savefig('steps.png')
    plt.close()

    plot_q_table(QTable, ACTIONS, 'q_table.png')

# Initialize GridWorld and Agent
MyGridWorld = GridWorld(GRID, START, END)
MyAgent = Agent(GRID)

steps = train(MyGridWorld, MyAgent)    # Train Agent
post_training(steps, MyAgent.q_table)    # Post Training
