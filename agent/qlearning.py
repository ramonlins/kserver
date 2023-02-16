from itertools import combinations

import numpy as np

from .greedy import GreedyAgent

class QlAgent(GreedyAgent):
    def __init__(self, env, gamma=0.9, alpha=0.01, epsilon=0.2, is_plot=False):
        self.env = env
        self.is_plot = is_plot
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.xdata = []
        self.ydata = []
        # For q-learning is necessary to know all state-action pair
        states = list(combinations(range(env.n), env.k))
        self.Q = {}
        self.create_Qtable(states)

    def policy(self, s):
        pi = np.ones(self.env.k, dtype=float) * self.epsilon/self.env.k
        s = tuple(s)
        greed_action = np.argmin(self.Q[s][0])
        pi[greed_action] = 1 - self.epsilon + pi[greed_action]
        a = np.random.choice(np.arange(len(pi)), p=pi)

        return a

    def update_Qtable(self, s, a, r, s_):
        # update Q table minimizing the cost
        Qtarget = r + self.gamma * min(self.Q[tuple(s_)][0])
        self.Q[tuple(s)][0][a] += self.alpha * (Qtarget - self.Q[tuple(s)][0][a])


    def create_Qtable(self, states):
        for state in states:
            self.Q[state] = np.zeros([1, self.env.k])
