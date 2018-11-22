# Deep-RL-agents

This repository contains the work I did during my Reinforcement Learning internship from September 2017 to February 2018.

During these 6 months, I reimplemented the main deep-RL algorithms that have been developped since 2013, using only Tensorflow and numpy.
This repository contains implementations of :
- [**A3C**][A3C] : the 2016 algorithm that uses asynchronous gradient descent for optimization on multi-CPU instead of a single GPU
- [**C51**][C51] : the 2017 algorithm that explores the idea of predicting not only the value of a state, but instead the value distribution
- [**DDPG**][DDPG] : the 2015 algorithm that tackles the problem of continuous control using an actor-critic architecture
- [**Rainbow**][Rainbow] : the 2017 algorithm that combines six classical extensions to DQN
- [**D4PG**][D4PG] : the 2018 algorithm that applies the distributional approach to a DDPG with an asynchronous architecture

The general architecture of these algorithm is always the same :
- the main.py file initialize the agent and run it
- the Model.py file implements the Neural Network (actor-critic or not, with convolution or not)
- the QNetwork.py file instantiates a Network and build the tensorflow operations to perform the gradient descent to train it
- the Agent.py file implements the agent class that interacts with the environment in order to get experiences
- the settings.py file is used to change the hyperparameters of the algorithm and the network

Others directories include :
- **utils** : a set of classes and functions used in other algorithms
- **BlogFiles** : a jupyter notebook that tries to explain the idea behind A3C, DDPG and Rainbow
- **Environment Test** : copies from the main algorithms set up to run in specific environments 
- **GIF** : a set of GIF saved after having trained different agents on many environments


[A3C]: https://arxiv.org/pdf/1602.01783.pdf
[C51]: https://arxiv.org/pdf/1707.06887.pdf
[DDPG]: https://arxiv.org/pdf/1509.02971.pdf
[Rainbow]: https://arxiv.org/pdf/1710.02298.pdf
[D4PG]: https://openreview.net/pdf?id=SyZipzbCb
