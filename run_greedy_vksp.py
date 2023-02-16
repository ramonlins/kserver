import time

import numpy as np
import matplotlib.pyplot as plt

from environment.vksp import VisualKspEnv
from agent.greedy import GreedyAgent
import analysis.plot as plot

print('Creating environment data ...')
start = time.time()

# reproducibility
#np.random.seed(0)

# n must be a perfect square
env = VisualKspEnv(n=100, k=3, T=1000, is_render=True)
agent = GreedyAgent(env)

reward_data, R = [], []
acc_rew_size = 100

# start agent-env interaction
for t in range(env.T):

    if env.is_render:
        env.render()

    a = agent.policy()

    o, r, done = env.step(a)

    # compute the accumulated reward
    reward_data.append(r)

    _, div = divmod(t, acc_rew_size)

    if div == 0 and t > 0:
        acc_rew = np.sum(reward_data)
        print('ep: {} reward (acc): {}'.format(t, acc_rew))
        if agent.is_plot:
            agent.plot_reward(t, acc_rew)
        reward_data.clear()
        R.append(acc_rew)

    if done:
        acc_rew = np.sum(reward_data)
        print('ep: {} reward (acc): {}'.format(t, acc_rew))
        if agent.is_plot:
            agent.plot_reward(t, acc_rew)
        R.append(acc_rew)


finish = time.time() - start
print('Running time {} sec'.format(finish))

plot.plot_accumulated_reward(R)
