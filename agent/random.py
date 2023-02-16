import numpy as np
import matplotlib.pyplot as plt

class RandomAgent:

    def __init__(self, action_space, is_plot=False):
        self.action_space = action_space
        self.xdata = []
        self.ydata = []
        self.is_plot = is_plot

        if self.is_plot:
            plt.subplots()

    def policy(self):
        a = np.random.randint(0, self.action_space)

        return a

    def plot_reward(self, x, y, color='r'):
        self.xdata.append(x)
        self.ydata.append(y)
        plt.plot(self.xdata, self.ydata, color)
        plt.draw()
        plt.pause(0.0000001)


