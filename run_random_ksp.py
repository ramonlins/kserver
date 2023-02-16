import random
import time

import numpy as np

from environment.ksp import KspEnv

# environment parameters
n = 10
k = n//2

#np.random.seed(0)

print('Creating environment data ...')
start = time.time()

env = KspEnv(n, k, T=10)
s = env.reset()

print('Agent interacting with the environment ...')
for t in range(env.T):
    env.render()

    a = random.randint(0, env.k-1)

    s, r, done = env.step(a)

    if done:
        break

finish = time.time() - start
print('Running time {} sec'.format(finish))
