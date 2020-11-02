import numpy as np
import scipy.linalg as LA
from lq.policies import LinK


class Linear_Quadratic:
    def __init__(self, A, B, Q, R, x0, ep):
        # self.random_seed = 1
        # np.random.seed(self.random_seed)
        self.A = A.astype('float32')
        self.B = B.astype('float32')
        self.Q = Q.astype('float32')
        self.R = R.astype('float32')
        self.n, self.m = B.shape
        self.ep = ep
        self.Qn = ep**2*np.eye(self.n)
        self.x0 = x0.astype('float32')
        self.x = x0
        self.P_opt, self.K_opt = self.OptK()

    def resetx(self):
        self.x = self.x0

    def randx(self):
        self.x = np.random.uniform(-1, 1, (1, self.n))

    def step(self, a):
        '''
        Step the environment
        Args:
            a: the action
        Returns:
            self.x: the next state
            c: the immediate cost
        '''
        c = self.x @ self.Q @ self.x.T + a @ self.R @ a.T
        self.x = self.x @ self.A.T + a @ self.B.T + self.ep * np.random.randn(1, self.n)
        return self.x, c

    def one_rollout(self, policy, T):
        '''
        Args:
            policy: The policy to do the rollout
            T: The rollout length
        Returns:
            states:
            actions:
            rewards:
            next_state:
        '''
        states = np.zeros((T, self.n), dtype='float32')
        actions = np.zeros((T, self.m), dtype='float32')
        rewards = np.zeros(T,  dtype='float32')
        next_states = np.zeros((T, self.n), dtype='float32')
        self.resetx()
        # self.randx()
        for t in range(T):
            states[t, :] = self.x
            actions[t, :] = policy(self.x)
            next_states[t, :], rewards[t] = self.step(actions[t, :])
        return states, actions, rewards, next_states

    def lqr_gain(self, given_q, given_r):
        '''
        lqr gain for the system
        :param given_q:
        :param given_r:
        :return: the kernel of the Lyapunov function P and the gain K
        '''
        try:
            P = LA.solve_discrete_are(self.A, self.B, given_q, given_r)
            K = -LA.inv(self.B.T @ P @ self.B + given_r) @ self.B.T @ P @ self.A
        except:
            P = np.zeros((self.n, self.n))
            K = np.zeros((self.m, self.n))
        return P, K

    def OptK(self):
        return self.lqr_gain(self.Q, self.R)

    def P_and_Pe_associated_to_K(self, K):
        if self.is_stable(K):
            cl_map = self.a_cl(K)
            P = LA.solve_discrete_lyapunov(cl_map.T, self.Q + K.T @ self.R @ K)
            distP = LA.norm(P - self.P_opt, 2) / LA.norm(self.P_opt, 2)
        else:
            P = 100.0 * np.eye(self.n)
            distP = float("inf")
        return P, distP

    def a_cl(self, K):
        return self.A + self.B @ K

    def is_stable(self, K):
        stab = False
        if np.amax(np.abs(LA.eigvals(self.a_cl(K)))) < (1.0 - 1.0e-6):
            stab = True
        return stab

    def dist_from_optimal_K(self, K):
        '''
        :param K: Given K
        :return: normalized L2 distance from K_opt
        '''
        if self.is_stable(K):
            distK = LA.norm(K - self.K_opt, 2) / LA.norm(self.K_opt, 2)
        else:
            distK = float("inf")
        return distK

    def cost_inf_K(self, K):
        '''
          Arguments:
            Control Gain K
            process noise covariance Qn
            observation noise covariance Rn
          Outputs:
            cost: Infinite time horizon LQR cost of static gain K
            u=Kx
        '''
        if self.is_stable(K):
            P,_ = self.P_and_Pe_associated_to_K(K)
            cost = np.trace(P @ self.Qn)
        else:
            cost = float("inf")

        return cost

    def cost_finite_average_K(self, K, T):
       '''
       :param K: The gain
       :param T: The horizon
       :return: Average Cost
       '''

       if self.is_stable(K):
           Lin_gain = LinK(K)
           _, _, cost, _= self.one_rollout(Lin_gain.lin_policy, T)
           ave_cost = np.sum(cost)/T
       else:
           ave_cost = 1000.0
       return ave_cost
