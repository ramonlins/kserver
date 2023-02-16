import matplotlib.pyplot as plt
import numpy as np

class GreedyAgent:

    def __init__(self, env, is_plot=False):
        self.env = env
        self.xdata = []
        self.ydata = []
        self.is_plot = is_plot

        if self.is_plot:
           plt.subplots()

    def policy(self):
        nj = self.env.req

        # find all servers costs to requisition
        costs = []
        for ni in self.env.s:
            costs.append(self.env.C[ni][nj])

        # action with min cost
        a = np.argmin(costs)

        return a

    def plot_reward(self, x, y, color='b'):
        self.xdata.append(x)
        self.ydata.append(y)
        plt.plot(self.xdata, self.ydata, color)
        plt.draw()
        plt.pause(0.0000001)

