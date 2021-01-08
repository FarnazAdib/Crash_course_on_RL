import numpy as np
from lq.pgrl import PGRL
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

## Making the PG agent and initializing storing matrices
# To be consistent among different methods, # n_iteation * nbatch should be the same for all algorithms
Mypgrl = PGRL(Mysys)
T = 10  # Rollout length
n_iteration = np.array([0, 500])  # An array for number of iterations
n_batch = 8  # Number of batches per iteration.
n_monte_carlo = 10  # Number of monte carlo evaluations.

## Initializing matrices
OPT = Storing_Mat_opt("Optimal Solution", Mysys, n_monte_carlo, len(n_iteration))
PG = Storing_Mat("Policy Gradient", Mysys, K0, n_monte_carlo, len(n_iteration))
Methods = [PG]

## Start Learning
for N in range(1, len(n_iteration)):
    for trial in range(n_monte_carlo):
        print('Number of iterations: %d, MC trial: %d' % (n_iteration[N], trial))

        print('Policy Gradient')
        PG.K[:, :, trial, N] = Mypgrl.pg_linpolicy(K0, n_iteration[N], n_batch, T, explore_mag=0.1)


for N in range(1, len(n_iteration)):
    for trial in range(n_monte_carlo):
        OPT.J_fin[trial, N] = Mysys.cost_finite_average_K(OPT.K, T)
        for meth in Methods:
            meth.J_inf[trial, N] = Mysys.cost_inf_K(meth.K[:, :, trial, N])
            meth.J_fin[trial, N] = Mysys.cost_finite_average_K(meth.K[:, :, trial, N], T)
            meth.Ke[trial, N] = Mysys.dist_from_optimal_K(meth.K[:, :, trial, N])

## Plotting the results
MyPlot = PLTLIB(OPT.J_inf)
J = [meth.J_inf for meth in Methods]
Lab = [meth.name for meth in Methods]
MyPlot.frac_stable(J, n_batch * n_iteration, n_monte_carlo, Lab, 'Number of rollouts')
MyPlot.est_e([meth.Ke for meth in Methods], n_batch * n_iteration, Lab, 'Number of rollouts', "Ke", 1)
MyPlot.relative_inf_cost(J, n_batch * n_iteration, Lab, 'Number of rollouts', uplim=0.3, zoomplt=False)
J.insert(0, OPT.J_inf * np.ones((n_monte_carlo, len(n_iteration))))
Lab.insert(0, OPT.name)
MyPlot.cost(J, n_batch * n_iteration, Lab, 'Number of rollouts', 'Infinite Cost', 0.1)

## Printing the median of estimated K
print('\nThe optimal k \n', OPT.K)
for meth in Methods:
    print('\nMean of estimated K using', meth.name, '\n', np.mean(meth.K[:, :, :, -1], axis=2))

for meth in Methods:
    print('\nMedian of Ke using', meth.name, np.median(meth.Ke[:, -1]))
