# 3rd party modules
import numpy as np


def run(m):
    """
        :param graph
        :return: shorter path for all nodes
    """
    l = m.shape[0]
    c = l
    dist = np.zeros((l, c))

    # initialization
    for i in range(l):
        for j in range(c):
            if i != j:
                dist[i][j] = np.inf
            else:
                dist[i][j] = 0

    # find minimum path for all nodes
    for i in range(l):
        h = dist[i]
        h = h.tolist()
        visited = []

        while True:
            d = np.nanmin(h)
            ind = h.index(d)
            for j in range(c):
                if j not in visited:
                    temp = d + m[ind][j]

                    if temp < h[j]:
                        h[j] = temp
                        dist[i][j] = h[j]

            visited.append(ind)
            h[ind] = np.nan
            if len(visited) == c:
                break

    return dist
