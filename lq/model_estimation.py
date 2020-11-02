import numpy as np
import scipy.linalg as LA
from lq.dynamics import Linear_Quadratic
from lq.policies import LinK
import scipy.odr.odrpack as odrpack
from lq.funlib import linear_func


class MODEL:
    def __init__(self, sysdyn:Linear_Quadratic):
        # self.rand_seed = 1
        # np.random.seed(self.rand_seed)
        self.dyn = sysdyn
        self.n, self.m = self.dyn.B.shape

    def nominal_AB(self, K, N, T, explore_mag=10, res_time=100):
        '''
        Estimates matrices A and B
        Args:
            K: The controller gain
            N: number of iterations to be consistent with iterative methods
            T: Rollout length
            explore_mag: noise magnitude for excitement
            res_time: when to reset the dynamics to its initial condition

        Returns: A_nom, B_nom
        '''
        Lin_gain = LinK(K)
        Lin_gain.make_sampling_on(explore_mag)
        if T >= res_time:
            N = int(np.ceil(N * T / res_time))
            T = res_time
        A_nom = np.zeros((self.n, self.n))
        B_nom = np.zeros((self.n, self.m))

        # storage matrices
        states_batch = np.zeros((self.n, N, T))
        next_states_batch = np.zeros((self.n, N, T))
        actions_batch = np.zeros((self.m, N, T))

        # simulate
        for k in range(N):
            # Do one rollout to save data for  model estimation
            states, actions, _, next_states = self.dyn.one_rollout(Lin_gain.sample_lin_policy, T)
            states_batch[:, k, :] = states.T
            actions_batch[:, k, :] = actions.T
            next_states_batch[:, k, :] = next_states.T

        for i in range(self.n):
            linear = odrpack.Model(linear_func)
            data = odrpack.RealData(
                x=np.vstack((states_batch.reshape(self.n, N * T), actions_batch.reshape(self.m, N * T))),
                y=next_states_batch[i, :, :].reshape(1, N * T))
            Myodr = odrpack.ODR(data, linear, np.zeros(self.n + self.m))
            out = Myodr.run()
            tmp = out.beta
            A_nom[i, :] = tmp[0:self.n]
            B_nom[i, :] = tmp[self.n:]

        return A_nom, B_nom

    def nominal_PK(self, K0, N, T, explore_mag=10, res_time=100):
        '''
        An iterative approach to find the optimal controller based on learning the dynamics.
        In each iteration, the system is rolled out with the current controller gain. Then, A_nom and B_nom
        are estimated and then the optimal controller gain K is computed by solving the Algebraic Riccati Equation
        Args:
            K0: Initial controller gain
            N: Number of iterations
            T: Rollout length
            explore_mag: Noise magnitude for excitement during data collection
            res_time: time to reset the initial condition of the system to have numerical stability

        Returns: P, K
        '''
        K = K0
        for n in range(N):
            if self.dyn.is_stable(K):
                A_nom, B_nom = self.nominal_AB(K, 1, T, explore_mag=explore_mag, res_time=res_time)
                try:
                    P = LA.solve_discrete_are(A_nom, B_nom, self.dyn.Q, self.dyn.R)
                    K = -LA.inv(B_nom.T @ P @ B_nom + self.dyn.R) @ B_nom.T @ P @ A_nom
                except:
                    P, K = self.unstable_P_and_K()
            else:
                P, K = self.unstable_P_and_K()
        return P, K

    def unstable_P_and_K(self):
        return np.zeros((self.n, self.n)), np.zeros((self.m, self.n))
