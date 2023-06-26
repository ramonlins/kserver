# Description
The k-Server Problem repository provides a comprehensive set of basic steps and tools for simulating the k-server problem. The k-server problem, a well-known optimization challenge, involves the efficient movement of servers to serve a sequence of requests without prior knowledge of future demands while minimizing overall costs such as time or distance.

In this repository, you will find a step-by-step guide and accompanying code that allows you to simulate the k-server problem. The simulation encompasses the fundamental aspects required to understand the problem and explore potential solutions. The following basic steps are covered:

- Problem Setup: Gain an understanding of the k-server problem by defining the problem parameters, including the number of servers, the number of nodes, and the sequence of requests. You will learn how to initialize the server positions and the initial state of the environment.

- Request Generation: Generate a sequence of requests based on a specified distribution or algorithm. This step ensures that the simulation accurately represents the dynamics and uncertainty of real-world scenarios.

- Server Movement: Implement the server movement strategy, which determines how the servers respond to each incoming request. Explore different server movements policies, such as random selection or strategies based on heuristics, to evaluate their impact on the overall performance.

- Cost Calculation: Calculate the cost associated with serving each request based on the server movements. The cost can be defined in various ways, such as the Euclidean distance traveled or the time taken to reach the requested node.

- Performance Evaluation: Assess the performance of the server movement strategy by analyzing the total cost incurred throughout the simulation. Compare different strategies and algorithms to identify the most effective approach for minimizing costs.

The repository provides example code and implementations in popular programming languages like Python, Java, or MATLAB, enabling researchers and practitioners to easily simulate and experiment with the k-server problem. Additionally, the code includes comments and documentation to facilitate understanding and modification.

By utilizing the resources in this repository, researchers and practitioners can gain hands-on experience in simulating the k-server problem and evaluating different strategies and algorithms. This serves as a foundation for further research and development of advanced optimization techniques to address this challenging problem.

Simulating the k-server problem is a crucial step in comprehending its dynamics and exploring potential solutions. By leveraging this repository, you will be equipped with the necessary tools and knowledge to advance your understanding of the k-server problem and contribute to the development of innovative algorithms and strategies in this field.

This repository creates the basic steps to simulate the k-server problem and test algorithms on top of it. The codes were based on the paper "[Deep reinforcement learning applied to the k-server problem](https://www.sciencedirect.com/science/article/abs/pii/S0957417419304154)" which proposes solving the k-server problem modeling the Q-function with a Multilayer Perceptron (MLP) neural network that perceives the dynamics of the problem through images that encode the environment. The results show that the new algorithm is more suitable for the curse of dimensionality problem.


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
