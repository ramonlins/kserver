# core packages
import numpy as np
import random

# personal modules
import environment.graph as graph

"""
state-action table enviroment
"""
class KspEnv:

    def __init__(self, n, k, T, c_max=100, sparsity=0.5, is_plot=False, is_log=True):
        self.n = n
        self.k = k
        self.T = T
        self.t = 0
        self.r = 0
        self.is_log = is_log
        self.log = list()
        self.temp = list()
        self.is_render = False
        self.initial_state()
        self.requisition()


        # calculate the minimum cost path matrix
        if is_plot:
            m, self.C = graph.rand(n, c_max, sparsity)
            graph.plot(m)
            print(self.C)
        else:
            _, self.C = graph.rand(n, c_max, sparsity)
            print(self.C)

    def step(self, a):

        # find server node and requisition node
        ni = self.s[a]
        nj = self.req
        self.r = self.C[ni][nj]

        # create a "log" file
        if self.is_log:
            temp = [self.s.copy(), self.req, self.r, a]
            self.log.append(temp)

        if self.is_render:
            print('s:{} req:{} reward:{} action:{}'.format(self.s.copy(), self.req, self.r, a))

        self.get_state(a)
        self.t = self.t + 1

        if self.t < self.T:
            is_done = False
        else:
            is_done = True

        return self.s, self.r, is_done

    def render(self):
        self.is_render = True

    def get_state(self, a):
        self.s[a] = self.req
        self.s.sort()
        self.requisition()

    def reset(self):
        self.initial_state()
        self.requisition()

    def requisition(self):
        self.req = random.randint(0, self.n-1)

        # requisition arrives in empty nodes
        while True:
            if self.req in self.s:
                self.req = random.randint(0, self.n-1)
            else:
                break

    def initial_state(self):
        nodes = list(range(0, self.n))
        random.shuffle(nodes)
        self.s = nodes[0:self.k]
        self.s.sort()
