import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import environment.graph as graph

"""state-action visual enviroment"""

class VisualKspEnv:

    def __init__(self,
                 nodes,
                 kservers,
                 episodes,
                 c_max=30,
                 sparsity=0.7,
                 is_plot=False,
                 is_log=True,
                 is_render=False):
        self.n = nodes
        self.k = kservers
        self.episodes = episodes
        self.t = 0
        self.r = 0
        self.is_log = is_log
        self.log = list()
        self.temp = list()
        self.h = int(np.round(np.sqrt(nodes)))
        self.w = self.h
        self.initial_state(None)
        self.requisition(None)
        self.color_model = 'rgb'
        self.is_render = is_render
        self.initial_s = self.s

        if is_render:
            self.fig, self.ax = plt.subplots()

        # calculate the minimum cost path matrix
        if is_plot:
            m, self.C = graph.rand(nodes, c_max, sparsity)
            graph.plot(m)

            fig, ax = plt.subplots()
            sns.heatmap(self.C, annot=True, ax=ax)
            plt.show()

        else:
            _, self.C = graph.rand(nodes, c_max, sparsity)
            #print(self.C)

    def step(self, a):
        # find server and requisition node
        ni = self.s[a]
        nj = self.req
        self.r = self.C[ni][nj]

        # create a "log" file
        if self.is_log:
            temp = [self.s.copy(), self.req, self.r, a]
            self.log.append(temp)

        o = self.get_state(a)  # next observation
        self.t = self.t + 1

        if self.t < self.episodes:
            is_done = False
        else:
            is_done = True
            self.t = 0

        return o, self.r, self.s, is_done

    def render(self):
        # print('s:{} req:{} reward:{} action:{}'.format(self.s.copy(), self.req, self.r, a))
        o = self.encode(self.s, self.req, self.color_model)  # present observation
        self.ax.cla()
        if self.color_model == 'rgb':
            self.ax.imshow(o)
        else:
            self.ax.imshow(o, cmap='gray')

        plt.pause(0.0000001)

    def get_state(self, a):
        self.s[a] = self.req
        self.s.sort()  # homogeneous server
        self.requisition(seed=None)

        o = self.encode(self.s, self.req, self.color_model)

        return o

    def encode(self, s, req, color_model):
        if color_model == 'rgb':
            data = np.arange(self.n).reshape(self.h, self.w)  # position matrix
            data2 = np.zeros((self.h, self.w, 3))  # encoded matrix

            for i in range(self.k):
                if s[i] in data:
                    # position of servers
                    ind = np.argwhere(data == s[i])
                    x = ind[0, 0]
                    y = ind[0, 1]

                    # all channels
                    data2[x][y][0] = 1
                    data2[x][y][1] = 1
                    data2[x][y][2] = 1

            # position of requisition
            ind = np.argwhere(data == req)
            x = ind[0, 0]
            y = ind[0, 1]
            data2[x][y][0] = 1  # red channel

        else:
            data = np.arange(self.n).reshape(self.h, self.w)
            data2 = np.zeros((self.h, self.w))

            for i in range(self.k):
                if s[i] in data:
                    ind = np.argwhere(data == s[i])
                    x = ind[0, 0]
                    y = ind[0, 1]
                    data2[x][y] = 1

            ind = np.argwhere(data == req)
            x = ind[0, 0]
            y = ind[0, 1]
            data2[x][y] = 0.5

        return data2

    # provide the servers distance to requisition as information on image
    def encode2(self, s, req, color_model):
        if color_model == 'rgb':
            data = np.arange(self.n).reshape(self.h, self.w)
            data2 = np.zeros((self.h, self.w, 3))

            for i in range(self.k):
                if s[i] in data:
                    # position of servers
                    ind = np.argwhere(data == s[i])
                    x = ind[0, 0]
                    y = ind[0, 1]

                    # all channels
                    data2[x][y][0] = self.C[s[i]][self.req]/np.amax(self.C)  # server distance information on image
                    data2[x][y][1] = self.C[s[i]][self.req]/np.amax(self.C)
                    data2[x][y][2] = self.C[s[i]][self.req]/np.amax(self.C)

            # position of requisition
            ind = np.argwhere(data == req)
            x = ind[0, 0]
            y = ind[0, 1]
            data2[x][y][0] = 1  # red channel

        else:
            data = np.arange(self.n).reshape(self.h, self.w)
            data2 = np.zeros((self.h, self.w))

            for i in range(self.k):
                if s[i] in data:
                    ind = np.argwhere(data == s[i])
                    x = ind[0, 0]
                    y = ind[0, 1]
                    data2[x][y] = (self.C[s[i]][self.req]/np.amax(self.C))*0.5

            ind = np.argwhere(data == req)
            x = ind[0, 0]
            y = ind[0, 1]
            data2[x][y] = 1

        return data2

    def reset(self, is_same_initial_state, seed=None):
        if is_same_initial_state and seed is not None:
            self.initial_state(seed)
            self.requisition(seed)
        else:
            self.initial_state(seed=None)
            self.requisition(seed=None)

        o = self.encode(self.s, self.req, self.color_model)

        return o, self.r, self.s, False

    def requisition(self, seed):
        if seed is not None:
            np.random.seed(0)

        self.req = np.random.randint(0, self.n)

        # requisition should arrives in empty node
        while True:
            if self.req in self.s:
                self.req = np.random.randint(0, self.n)
            else:
                break

    def initial_state(self, seed):
        if seed is not None:
            np.random.seed(0)

        nodes = list(range(0, self.n))
        np.random.shuffle(nodes)
        s = nodes[0:self.k]
        s.sort()
        self.s = s
