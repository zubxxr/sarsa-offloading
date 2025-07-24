import random
from collections import defaultdict

class SARSAAgent:
    def __init__(self, epsilon=1.0, alpha=0.1, gamma=0.9, min_epsilon=0.1, decay_rate=0.995):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.actions = ["local", "cloud", "edge1", "edge2"]

    def discretize_state(self, battery, cpu, bandwidth):
        return (
            round(battery / 10) * 10,
            round(cpu / 10) * 10,
            round(bandwidth / 10) * 10,
        )

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_vals = self.q_table[state]
        return max(q_vals, key=q_vals.get) if q_vals else random.choice(self.actions)

    def update(self, state, action, reward, next_state, next_action):
        predict = self.q_table[state][action]
        target = reward + self.gamma * self.q_table[next_state][next_action]
        self.q_table[state][action] += self.alpha * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)