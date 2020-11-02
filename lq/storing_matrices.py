import numpy as np
from lq.dynamics import Linear_Quadratic

class Storing_Mat():
    def __init__(self, name, sysdyn:Linear_Quadratic, K0, n_mc, T):
        self.name = name
        self.n_mc = n_mc
        self.T = T
        self.J_inf = np.zeros((n_mc, T))
        self.J_fin = np.zeros((n_mc, T))
        self.K = np.zeros([sysdyn.m, sysdyn.n, n_mc, T])
        self.Ke = np.zeros([n_mc, T])
        self.P = np.zeros([sysdyn.n, sysdyn.n, n_mc, T])
        self.Pe = np.zeros([n_mc, T])
        for trial in range(self.n_mc):
            self.K[:, :, trial, 0] = K0
            self.J_fin[trial, 0] = sysdyn.cost_finite_average_K(K0, T)
            self.J_inf[trial, 0] = sysdyn.cost_inf_K(K0)
            self.Ke[trial, 0] = sysdyn.dist_from_optimal_K(K0)


class Storing_Mat_opt():
    def __init__(self, name, sysdyn:Linear_Quadratic, n_mc, T):
        self.name = name
        self.n_mc = n_mc
        self.T = T
        self.J_fin = np.zeros((n_mc, T))
        self.P = sysdyn.P_opt
        self.K = sysdyn.K_opt
        self.J_inf = sysdyn.cost_inf_K(self.K)
        for trial in range(self.n_mc):
            self.J_fin[trial, 0] = sysdyn.cost_finite_average_K(self.K, T)
