import time

import numpy as np

from environment.vksp import VisualKspEnv
from agent.random import RandomAgent
import analysis.plot as analysis_plot

print('Creating environment data ...')
start = time.time()

# reproducibility
np.random.seed(0)

# n must be a perfect square
env = VisualKspEnv(n=9, k=2, T=1000, is_render=True)
agent = RandomAgent(env.k, is_plot=True)

# analysis parameters
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

    if div == 0:
        acc_rew = sum(reward_data)
        print('ep: {} reward (mean): {}'.format(t, acc_rew))
        if agent.is_plot:
            agent.plot_reward(t, acc_rew)
        reward_data.clear()

    if done:
        acc_rew = np.sum(reward_data)
        print('ep: {} reward (mean): {}'.format(t, acc_rew))
        if agent.is_plot:
            agent.plot_reward(t, acc_rew)
        R.append(acc_rew)


finish = time.time() - start
print('Running time {} sec'.format(finish))

analysis_plot.plot_accumulated_reward(R)
