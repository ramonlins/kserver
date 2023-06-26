# Description
Posed more than three decades ago, the k-server problem is characterized by the necessity to the displacement of a server to attend a requisition sequence through time without prior knowledge of future demands. The goal is to minimize the total cost (i.e. distance, time) involved in the process. Although conceptually simple, its complexity grows exponentially with the increase in the number of nodes and servers. Besides, after a decision is made, it cannot be revoked, influencing the solution as a whole. The k-server problem is still an open problem serving as fuel for the development of new algorithms.

As an alternative solution, a reinforcement learning algorithm was presented at IEEE International Joint Conference on Neural Networks (IJCNN), 2005. The results showed that the reinforcement learning approach could achieve better performance than influential algorithms of the literature. Despite this, the approach was based on the Q-learning algorithm, subject to the curse of dimensionality problem, since its storage structure (Q table), grows exponentially as the number of states and actions increases.

This repository creates de basic steps to simulate the k-server problem and test algorithms on top of it. The codes are based on the paper "[Deep reinforcement learning applied to the k-server problem](https://www.sciencedirect.com/science/article/abs/pii/S0957417419304154)" which proposes solving the k-server problem modeling the Q-function with a Multilayer Perceptron (MLP) neural network that perceives the dynamics of the problem through images that encode the environment. The results show that the new algorithm is more suitable for the curse of dimensionality problem.

# Requirements
To create an environment:
```bash
conda env create -f environment.yml
```

# Get start
To test agents you can run:
```bash
python run_greedy_vksp.py
```
