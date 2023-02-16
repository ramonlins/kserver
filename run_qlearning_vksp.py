# 3rd party modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# core modules
import time

# personal modules
from environment.vksp import VisualKspEnv
from agent.qlearning import QlAgent
import analysis.plot as plot

print('Creating environment data ...')
start = time.time()

# reproducibility
#np.random.seed(0)

# graph parameters
n = 9
k = 3

# rl parameters
epochs = 1000
is_same_initial_state = True
seed = None # Random initial states
decay = 0.1
patient = 10

# n must be a perfect square
env = VisualKspEnv(nodes=n,
                   kservers=k,
                   episodes=epochs,
                   is_render=False,
                   c_max=10,
                   sparsity=0.1,
                   is_plot=True)

agent = QlAgent(env)

reward_data, R = [], []
best_cost = np.inf
# start agent-env interaction
for epoch in range(epochs):
    _, r, s, done = env.reset(is_same_initial_state, seed)

    reward_data = []
    while not done:
        if env.is_render:
            env.render()

        # choose action
        a = agent.policy(s)

        # take action
        _, r, s_, done = env.step(a)

        # update Q table
        agent.update_Qtable(s, a, r, s_)

        # update state
        s = s_

        # compute the accumulated reward
        reward_data.append(r)

        agent.epsilon -= decay * agent.epsilon

    if done:
        acc_rew = np.sum(reward_data)
        print('ep: {} cost: {}'.format(epoch, acc_rew))
        if agent.is_plot:
            agent.plot_reward(epoch, acc_rew)
        R.append(acc_rew)

        if acc_rew < best_cost and epoch > patient:
            print(f"best score {acc_rew}")
            best_cost = acc_rew
            agent.epsilon *= agent.epsilon * 2

finish = time.time() - start
print('Running time {} sec'.format(finish))

plot.plot_accumulated_reward(R)

print(agent.Q)
