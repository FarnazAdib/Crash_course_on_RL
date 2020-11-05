# A Crash Course on Reinforcement Learning for Control Problems Using TensorFlow 2

This is a self-contained repository to explain two basic Reinforcement (RL) algorithms, namely __Policy Gradient (PG)__ and __Q-learning__, and show how to apply them on control problems. Dynamical systems might have discrete action-space like cartpole where two possible actions are +1 and -1 or continuous action space like linear Gaussian systems. Usually, you can find a code for only one of these cases. It might be not obvious how to extend one to another. 

In this repository, I will explain how to formulate PG and Q-learning for each of these cases. I will provide implementations for these algorithms for both cases as Jupyter notebooks. You can also find the pure code for these algorithms (and also a few more algorithms that I have implemented by not discussed). The code is easy to follow and read. I have written in a modular way, so for example, if one is interested in the implementation of an algorithm is not confused with defining an environment in gym or plotting the results or so on.  

## Citing this repo
If you use this repo, please consider citing the following papers.
* F. Adib Yaghmaie, S. Gunnarsson and F. L. Lewis ["Output Regulation of Unknown Linear Systems using Average Cost Reinforcement Learning"](https://www.sciencedirect.com/science/article/pii/S0005109819304108), _Automatica_, Vol. 110, 2019.

* F. Adib Yaghmaie and F. Gustafsson ["Using Reinforcement Learning for Model-free Linear Quadratic Control with Process and Measurement Noises"](https://ieeexplore.ieee.org/abstract/document/9029904), _In 2019 Decision and Control (CDC)4, IEEE 58th Conference on,
2019_, pp. 6510-6517.

## How to start

Before starting running (and playing) with these algorithms, make sure you have built a virtual environment and imported required libraries. Take a look at [Preparation notebook](Preparation.ipynb).

## Dynamical systems cartple, 
We consider two types of dynamical systems (or environments in RL terminology). Read about them here

* [Cartpole: an environment with discrete action-space](cartpole.ipynb)
* [Linear Gaussian: an environment with continuous action space](linear_quadratic.ipynb)

## Policy Gradient

Below, you can see __jupyter notebooks__ regarding Policy Gradient (PG) algorithm

* [Explanation of Policy Gradient (PG)](pg_notebook.ipynb)
    * [How to code PG for problems with discrete action space (cartpole)](pg_on_cartpole_notebook.ipynb)
    * [How to code PG for problems with continuous action space (linear quadratic)](pg_on_lq_notebook.ipynb)
    
You can also see the __pure code__ for PG
* PG pure code
    * [PG for discrete action space (cartpole)](./cartpole/pg_on_cartpole.py)
    * [PG for continuous action space (linear quadratic)](./lq/pg_on_lq.py)

## $Q$-learning 

Below, you can see __jupyter notebooks__ regarding $Q$-learning algorithm
* [Explanation of $Q$-learning](q_notebook.ipynb)
    * [How to code $Q$-learning for problems with discrete action space (cartpole)](q_on_cartpole_notebook.ipynb)
    * [How to code $Q$-learing for problems with continuous action space (linear quadratic)](q_on_lq_notebook.ipynb)
* [Explanation of experience replay $Q$-learning](replay_q_notebook.ipynb)
    * [How to code experience replay $Q$ learning for systems with discrete action space (cartple)](replay_q_on_cartpole_notebook.ipynb)
    * _We have not implemented explerience replay $Q$ learning on LQ problem because the plain $Q$-learning is super good on LQ porblem. Note that as you can see from the explanation in the experience replay $Q$-learning, this algorithm has only two simple functions in addition to the plain $Q$-learning and those are not related to the action to be discrete or continuous. So, the extension is quite straight forward._
    
You can also see the __pure code__ for $Q$- and experience replay $Q$-learning
    
* $Q$-learning pure code
    * [$Q$-learning for discrete action space (cartpole)](./cartpole/q_on_cartpole.py)
    * [$Q$-learning for continuous action space (linear quadratic)](./lq/q_on_lq.py)
    
* Experience replay $Q$-learning pure code
    * [Experience replay $Q$ learning for discrete action space (cartple)](./cartpole/replay_q_on_cartpole.py)