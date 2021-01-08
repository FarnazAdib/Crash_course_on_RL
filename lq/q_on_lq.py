import numpy as np
from lq.policy_iteration import Q_learning, off_policy_learning
from lq.model_estimation import MODEL
from lq.dynamics import Linear_Quadratic
from lq.storing_matrices import Storing_Mat, Storing_Mat_opt
from lq.pltlib import PLTLIB
## Making the environment
rand_seed = 1
np.random.seed(rand_seed)
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
x0 = np.array([[-1.0, 0.0]], dtype='float32')
ep = 0.1
Q = np.array([[1, 0], [0, 1]])
R = np.array([[1]])
Mysys = Linear_Quadratic(A, B, Q, R, x0, ep)
_, K0 = Mysys.lqr_gain(200 * Mysys.Q, Mysys.R)

## Making the policy iteration agent
My_q_learning = Q_learning(Mysys)
My_model_learning = MODEL(Mysys)
T = np.array([0, 100])
n_iteration = 5
n_monte_carlo = 10

## Initializing matrices
OPT = Storing_Mat_opt("Optimal Solution", Mysys, n_monte_carlo, len(T))
Q = Storing_Mat("Q-learning", Mysys, K0, n_monte_carlo, len(T))
MODEL_LEARNING = Storing_Mat("Model-learning", Mysys, K0, n_monte_carlo, len(T))
Methods = [Q, MODEL_LEARNING]

## Start Learning
for t in range(1, len(T)):
    for trial in range(n_monte_carlo):
        print('Rollout Length: %d, MC trial: %d' % (T[t], trial))

        print('Q learning')
        Q.P[:, :, trial, t], Q.K[:, :, trial, t] = My_q_learning.ql(K0, n_iteration, T[t], explore_mag=1.0)

        print('Model learning')
        MODEL_LEARNING.P[:, :, trial, t], MODEL_LEARNING.K[:, :, trial, t] = \
            My_model_learning.nominal_PK(K0, n_iteration, T[t])

for t in range(1, len(T)):
    for trial in range(n_monte_carlo):
        OPT.J_fin[trial, t] = Mysys.cost_finite_average_K(OPT.K, T[t])
        for meth in Methods:
            meth.J_inf[trial, t] = Mysys.cost_inf_K(meth.K[:, :, trial, t])
            meth.J_fin[trial, t] = Mysys.cost_finite_average_K(meth.K[:, :, trial, t], T[t])
            meth.Ke[trial, t] = Mysys.dist_from_optimal_K(meth.K[:, :, trial, t])
            _, meth.Pe[trial, t] = Mysys.P_and_Pe_associated_to_K(meth.K[:, :, trial, t])
## Plotting the results
MyPlot = PLTLIB(OPT.J_inf)
J = [meth.J_inf for meth in Methods]
Lab = [meth.name for meth in Methods]
MyPlot.frac_stable(J, T, n_monte_carlo, Lab, 'Rollout Length')
MyPlot.est_e([meth.Ke for meth in Methods], T, Lab, 'Rollout Length', "Ke", 1)
MyPlot.relative_inf_cost(J, T, Lab, 'Rollout Length', uplim=0.3, zoomplt=False)
J.insert(0, OPT.J_inf*np.ones((n_monte_carlo, len(T))))
Lab.insert(0, OPT.name)
MyPlot.cost(J, T, Lab, 'Rollout Length', 'Infinite Cost', 0.1)

## Printing the median of estimated K
print('\nThe optimal k \n', OPT.K)
for meth in Methods:
    print('\nMean of estimated K using', meth.name, '\n', np.mean(meth.K[:, :, :, -1], axis=2))

for meth in Methods:
    print('\nMedian of Ke using', meth.name, np.median(meth.Ke[:, -1]))
