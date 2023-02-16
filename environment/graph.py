# 3rd party modules
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# personal modules
import environment.dijkstra as dijkstra


# create a connected random graph
def rand(n, max_weight, density):
    # create a random adjency matrix
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = np.random.random()
                if p < density:  # kind of sparsity
                    m[i][j] = np.random.randint(1, max_weight + 1)
                    m[j][i] = m[i][j]
                else:
                    m[i][j] = np.inf
                    m[j][i] = m[i][j]

    # connect isolated nodes
    for i, row in enumerate(m):
        connected = check(row)  # check if is connected
        if not connected:
            while True:
                for j in range(n):
                    if i != j:
                        p = np.random.random()
                        if p < 0.9:  # kind of sparsity
                            m[i][j] = np.random.randint(1, max_weight + 1)
                            m[j][i] = m[i][j]

                row = m[i][:]
                connected = check(row)
                if connected:
                    break
    c = dijkstra.run(m)

    return m, c


# check connectivity
def check(row):
    brow = np.isinf(row)
    edges = np.count_nonzero(~brow) - 1  # number of edges

    if edges > 0:
        connected = True
    else:
        connected = False

    return connected


def manual():
    m = np.array([
        [0, 3, np.inf, np.inf],
        [3, 0, 5, 2],
        [np.inf, 5, 0, 6],
        [np.inf, 2, 6, 0]])

    c = dijkstra.run(m)

    return m, c


def plot(m):
    # take off inf values
    for i, row in enumerate(m):
        for j, element in enumerate(row):
            if element == np.inf:
                m[i][j] = 0

    g = nx.Graph(m)
    nx.draw_networkx(g, with_labels=False, alpha=0.5)
    plt.show()
