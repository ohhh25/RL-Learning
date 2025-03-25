import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.GridWorld import GridWorld, get_world, get_random_world
from utils.q_table import plot_q_table

# Arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--world', type=int, default=0)

# Constants
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
if (argparser.parse_args().world == -1):
    GRID, START, END = get_random_world()
else:
    GRID, START, END = get_world(argparser.parse_args().world)

MAX_STEPS = GRID.shape[0] * GRID.shape[1] * 10

# Hyperparameters
learning_rate = 0.1
episodes = 10000

class Agent:
    def __init__(self, grid, epsilon=1.0, epsilon_decay=0.99, min_epsilon=1e-3,
                 discount=0.999):
        n_rows, n_cols = grid.shape
        self.q_table = np.zeros((n_rows, n_cols, 4))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.discount = discount

    def get_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon and train:
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
        while grid_world.get_cell() != 2:
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

def main():
    # Initialize GridWorld and Agent
    MyGridWorld = GridWorld(GRID, START, END)
    MyGridWorld.visualize('GridWorlds/simple.png')
    MyAgent = Agent(GRID)

    steps = train(MyGridWorld, MyAgent)    # Train Agent
    post_training(steps, MyAgent.q_table)    # Post Training
    MyGridWorld.create_path(MyAgent)    # Create Path

if __name__ == '__main__':
    main()
