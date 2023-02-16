import time

import environment.graph as graph

if __name__ == '__main__':

    print('Creating environment data')
    start = time.time()

    # Define graph
    nodes = 10; weigths = 2; density = 0.5

    # Generate adjacency matrix and the cost matrix from random graph
    adj, cost = graph.rand(nodes, weigths, density)

    print(f"ADJACENCY MATRIX \n{adj}"),

    print(f"COST MATRIX \n{cost}")

    finish = time.time() - start
    print('Running time {} sec'.format(finish))

    graph.plot(adj)
