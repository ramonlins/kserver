# 3rd party modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# core modules
from itertools import combinations

s_index_freq = []


def histogram_states(visited, n, k, data_name="", data_color="#539caf", x_label="", y_label="", title=""):
    comb = combinations(range(n), k)
    states = list(comb)
    n_bins = len(states)

    for s in visited:
        s_index = states.index(tuple(s))
        s_index_freq.append(s_index)

    # create the plo
    _, ax = plt.subplots()
    ax.hist(s_index_freq, bins=n_bins, color=data_color, alpha=0.75, label=data_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.show()


def plot_accumulated_reward(r):
    R = np.asarray(r)
    plt.subplots()
    sns.lineplot(data=R, errorbar='sd')
    plt.show()
