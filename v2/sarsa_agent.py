import random
import numpy as np

class SARSAAgent:
    def __init__(self, battery_bins=5, cpu_bins=5, bw_bins=5, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.battery_bins = battery_bins
        self.cpu_bins = cpu_bins
        self.bw_bins = bw_bins

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

        self.actions = ['local', 'edge1', 'edge2', 'cloud']
        self.q_table = {}

    def discretize_state(self, battery, cpu, bw):
        # Convert continuous values to discrete bins
        b = int(battery / (100 / self.battery_bins))
        c = int(cpu / (100 / self.cpu_bins))
        w = int(bw / (10 / self.bw_bins))
        return (b, c, w)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_vals = [self.get_q(state, a) for a in self.actions]
            max_index = np.argmax(q_vals)
            return self.actions[max_index]

    def update(self, state, action, reward, next_state, next_action):
        current_q = self.get_q(state, action)
        next_q = self.get_q(next_state, next_action)
        td_target = reward + self.gamma * next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.q_table[(state, action)] = new_q
