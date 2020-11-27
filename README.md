# A Crash Course on Reinforcement Learning for Control Problems Using TensorFlow 2

This is a self-contained repository to explain two basic Reinforcement (RL) algorithms, namely __Policy Gradient (PG)__ and __Q-learning__, and show how to apply them on control problems. Dynamical systems might have discrete action-space like cartpole where two possible actions are +1 and -1 or continuous action space like linear Gaussian systems. Usually, you can find a code for only one of these cases. It might be not obvious how to extend one to another. 

In this repository, I will explain how to formulate PG and Q-learning for each of these cases. I will provide implementations for these algorithms for both cases as Jupyter notebooks. You can also find the pure code for these algorithms (and also a few more algorithms that I have implemented but not discussed). The code is easy to follow and read. I have written in a modular way, so for example, if one is interested in the implementation of an algorithm is not confused with defining an environment in gym or plotting the results or so on.  

## Citing this repo
You can cite this repo

```
@misc{FarnazRLcourse,
  author = {Adib Yaghmaie, Farnaz},
  title = {Crash Course on Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FarnazAdib/Crash_course_on_RL.git}},
}
```


If you use this repo, please consider citing the following relevant papers:
* F. Adib Yaghmaie, S. Gunnarsson and F. L. Lewis ["Output Regulation of Unknown Linear Systems using Average Cost Reinforcement Learning"](https://www.sciencedirect.com/science/article/pii/S0005109819304108), _Automatica_, Vol. 110, 2019.

* F. Adib Yaghmaie and F. Gustafsson ["Using Reinforcement Learning for Model-free Linear Quadratic Control with Process and Measurement Noises"](https://ieeexplore.ieee.org/abstract/document/9029904), _In 2019 Decision and Control (CDC)4, IEEE 58th Conference on,
2019_, pp. 6510-6517.

* F. Adib Yaghmaie and s. Gunnarsson ["A New Result on Robust Adaptive Dynamic Programming for Uncertain Partially Linear Systems"](https://ieeexplore.ieee.org/abstract/document/9029695/), _In 2019 Decision and Control (CDC)4, IEEE 58th Conference on,
2019_, pp. 7480-7485.

## How to use this repo
This repository contains Jupyter notebooks and python files. If you want to run Jupyter notebooks, I suggest to use google colab. If you want to extend the results, examine more systems, I suggest to clone this repostory and run on your computer

### Running on google colab
* Go to [https://colab.research.google.com/notebooks/intro.ipynb] and sign in with a Google acount.
* Click File, and Upload notebook
* Select github and paste the following link [https://github.com/FarnazAdib/Crash_course_on_RL.git]
* Open the notebook that you want

### Running on local computer
* Go to [https://github.com/FarnazAdib/Crash_course_on_RL.git] and clone the project.
* Open PyCharm. From PyCharm. Click File and open project. Then, navigate to the project folder.
* Follow [Preparation notebook](Preparation.ipynb) to build a virtual environment and import required libraries.



## Where to start

You can start by reading about Reinforcement Learning
* [An introduction to Reinforcement Learning](RL_Intro.ipynb)


### Dynamical systems
You can read about dynamics systems (or environments in RL terminology) that we consider in this repo here.
* [Cartpole: an environment with discrete action-space](cartpole.ipynb)
* [Linear Gaussian: an environment with continuous action space](linear_quadratic.ipynb)

### Policy Gradient
Policy Gradient is one of the popular RL routines that relies upon optimizing the policy directly. Below, you can see __jupyter notebooks__ regarding Policy Gradient (PG) algorithm

* [Explanation of Policy Gradient (PG)](pg_notebook.ipynb)
    * [How to code PG for problems with discrete action space (cartpole)](pg_on_cartpole_notebook.ipynb)
    * [How to code PG for problems with continuous action space (linear quadratic)](pg_on_lq_notebook.ipynb)
    
You can also see the __pure code__ for PG
* PG pure code
    * [PG for discrete action space (cartpole)](./cartpole/pg_on_cartpole.py)
    * [PG for continuous action space (linear quadratic)](./lq/pg_on_lq.py)

### Q-learning 
Q-learning is another popular RL routine that relies upon dynamic programming. Below, you can see __jupyter notebooks__ regarding Q-learning algorithm
* [Explanation of Q-learning](q_notebook.ipynb)
    * [How to code Q-learning for problems with discrete action space (cartpole)](q_on_cartpole_notebook.ipynb)
    * [How to code Q-learing for problems with continuous action space (linear quadratic)](q_on_lq_notebook.ipynb)
* [Explanation of experience replay Q-learning](replay_q_notebook.ipynb)
    * [How to code experience replay Q learning for systems with discrete action space (cartple)](replay_q_on_cartpole_notebook.ipynb)
    * _We have not implemented explerience replay Q learning on LQ problem because the plain Q-learning is super good on LQ porblem. Note that as you can see from the explanation in the experience replay Q-learning, this algorithm has only two simple functions in addition to the plain Q-learning and those are not related to the action to be discrete or continuous. So, the extension is quite straight forward._
    
You can also see the __pure code__ for Q- and experience replay Q-learning
    
* Q-learning pure code
    * [Q-learning for discrete action space (cartpole)](./cartpole/q_on_cartpole.py)
    * [Q-learning for continuous action space (linear quadratic)](./lq/q_on_lq.py)
    
* Experience replay Q-learning pure code
    * [Experience replay Q-learning for discrete action space (cartple)](./cartpole/replay_q_on_cartpole.py)