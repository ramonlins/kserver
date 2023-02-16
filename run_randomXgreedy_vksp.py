# 3rd party modules
import numpy as np
import matplotlib.pyplot as plt

# core modules
import time
from copy import copy

# personal modules
import environment.vksp as vksp
import agent.random as random
import agent.greedy as greedy

import analysis.plot as plot

print('Creating environment data ...')
start = time.time()

# reproducibility
np.random.seed(0)

# n must be a perfect square
env1 = vksp.VisualKspEnv(n=9, k=4, T=1000, is_render=True)
env2 = copy(env1)

is_plot = True
agent1 = random.RandomAgent(env1.k, is_plot=is_plot)
agent2 = greedy.GreedyAgent(env2, is_plot=is_plot)


reward_data1, R1 = [], []
reward_data2, R2 = [], []
acc_rew_size = 100

# start agent-env interaction
for t in range(env1.T):

    if env1.is_render:
        env1.render()

    a1 = agent1.policy()
    o1, r1, done1 = env1.step(a1)
    reward_data1.append(r1)

    if env2.is_render:
        env2.render()

    a2 = agent2.policy()
    o2, r2, done2 = env2.step(a2)
    reward_data2.append(r2)

    # compute the accumulated reward

    _, div = divmod(t, acc_rew_size)

    if div == 0 and t > 0:
        acc_rew1 = sum(reward_data1)
        print('ep: {} acc_reward (rand): {}'.format(t, acc_rew1))
        if agent1.is_plot:
            agent1.plot_reward(t, acc_rew1)
        reward_data1.clear()
        R1.append(acc_rew1)

        acc_rew2 = sum(reward_data2)
        print('ep: {} acc_reward (greedy): {}'.format(t, acc_rew2))
        if agent2.is_plot:
            agent2.plot_reward(t, acc_rew2)
        reward_data2.clear()
        R2.append(acc_rew2)

    if done1:
        acc_rew1 = np.sum(reward_data1)
        print('ep: {} acc_reward (rand): {}'.format(t, acc_rew1))
        if agent1.is_plot:
            agent1.plot_reward(t, acc_rew1)
        R1.append(acc_rew1)

        acc_rew2 = np.sum(reward_data2)
        print('ep: {} acc_reward (greedy): {}'.format(t, acc_rew2))
        if agent2.is_plot:
            agent2.plot_reward(t, acc_rew2)
        R2.append(acc_rew2)


finish = time.time() - start
print('Running time {} sec'.format(finish))

plot.plot_accumulated_reward(R1)
plot.plot_accumulated_reward(R2)

