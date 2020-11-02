import numpy as np
import scipy.linalg as LA
from lq.dynamics import Linear_Quadratic
from lq.policies import LinK
from lq.funlib import GtoP, vecv, SquareMat, inst_variable


class Q_learning:
    def __init__(self, sysdyn:Linear_Quadratic):
        # self.rand_seed = 1
        # np.random.seed(self.rand_seed)
        self.dyn = sysdyn
        self.n, self.m = self.dyn.B.shape
        self.n_phi = int((self.n + self.m) * (self.n + self.m + 1) / 2)
        self.P = np.zeros((self.n, self.n))

    def ql(self, K0, N, T, explore_mag=1.0):
        '''
        Q learning loop to iterate over policy iteration and policy improvement
        :param K0: The initial policy gain
        :param N: Number of iterations
        :param T: Trajectory length
        :param explore_mag: The amount of randomness in Q learning
        :return: The kernel of the value function P and the controller gain K
        '''
        self.K = K0
        for k in range(N):

            # If the controller is stable, do an iteration
            if self.dyn.is_stable(self.K):

                # Policy evaluaion
                G = self.q_evaluation(T, explore_mag)

                # Policy improvement
                self.K = self.q_improvement(G)
                P = GtoP(G, self.K)

            # If the controller is not stable, return some unstable values for P and K
            else:
                P, self.K = self.unstable_P_and_K()

        return P, self.K

    def q_evaluation(self, T, explore_mag):
        # creating the linear policy and turning sampling on
        Lin_gain = LinK(self.K)
        Lin_gain.make_sampling_on(explore_mag)

        # Do one rollout to compute the average cost
        _, _, r, _ = self.dyn.one_rollout(Lin_gain.lin_policy, T)
        Lam = np.sum(r)/T

        # Do one rollout to save data for Q learning
        states, actions, costs, next_states = self.dyn.one_rollout(Lin_gain.sample_lin_policy, T)

        # Making the state z and the next state z
        z = np.concatenate((states, actions), axis=1)
        next_z = np.concatenate((next_states, Lin_gain.lin_policy(next_states)), axis=1)

        # estimating the Q parameter using instrumental variable
        x_iv = vecv(z) - vecv(next_z)
        y_iv = costs - Lam
        z_iv = vecv(z)
        q_vec = inst_variable(x_iv, y_iv, z_iv)
        return SquareMat(q_vec, self.n+self.m)

    def q_improvement(self, G):
        return - LA.inv(G[self.dyn.n:, self.dyn.n:]) @ G[self.dyn.n:, 0:self.dyn.n]

    def unstable_P_and_K(self):
        return np.zeros((self.n, self.n)), np.zeros((self.m, self.n))

class off_policy_learning:
    def __init__(self, sysdyn:Linear_Quadratic):
        # self.rand_seed = 1
        # np.random.seed(self.rand_seed)
        self.dyn = sysdyn
        self.n, self.m = self.dyn.B.shape
        self.n_phi = int((self.n + self.m) * (self.n + self.m + 1) / 2)
        self.P = np.zeros((self.n, self.n))

    def unstable_P_and_K(self):
        return np.zeros((self.n, self.n)), np.zeros((self.m, self.n))

    def off_policyl(self, K0, N, T, explore_mag=1.0):
        self.K = K0
        for k in range(N):
            if self.dyn.is_stable(self.K):
                P, BtPA, BtPB = self.off_policy_evaluation(T, explore_mag)
                self.K = self.off_policy_improvement(BtPB, BtPA)
            else:
                P, self.K = self.unstable_P_and_K()
        return P, self.K

    def off_policy_evaluation(self, T, explore_mag):
        # creating the linear policy and turning sampling on
        Lin_gain = LinK(self.K)
        Lin_gain.make_sampling_on(explore_mag)

        # Do one rollout to compute the average cost
        _, _, r, _ = self.dyn.one_rollout(Lin_gain.lin_policy, T)
        Lam = np.sum(r) / T

        # Do one rollout to save data for off-policy learning
        states, actions, costs, next_states = self.dyn.one_rollout(Lin_gain.sample_lin_policy, T)

        # estimating the parameter using instrumental variable
        y_iv = np.zeros(T)
        z_iv2 = np.zeros((T, self.n * self.m))
        actions_target = Lin_gain.lin_policy(states)
        for t in range(T):
            y_iv[t] = states[t, :] @ self.dyn.Q @ states[t, :].T \
                      + actions_target[t, :] @ self.dyn.R @ actions_target[t, :].T - Lam
            z_iv2[t, :] = 2 * LA.kron(states[t, :].reshape(1, self.n),
                                      (actions[t, :] - actions_target[t, :]).reshape(1, self.m))
        z_iv1 = vecv(states)
        z_iv3 = vecv(actions) - vecv(actions_target)
        z_iv = np.concatenate((z_iv1, z_iv2, z_iv3), axis=1)
        x_iv = np.concatenate((z_iv1-vecv(next_states), z_iv2, z_iv3), axis=1)
        W = inst_variable(x_iv, y_iv, z_iv)

        nP = int(self.n * (self.n + 1) / 2)
        P = SquareMat(W[0:nP], self.n)
        BtPA = W[nP:nP + self.m * self.n].reshape([self.m, self.n], order='F')
        BtPB = SquareMat(W[nP + self.m * self.n:], self.m)
        return P, BtPA, BtPB

    def off_policy_improvement(self, BtPB, BtPA):
        return - LA.inv(BtPB + self.dyn.R) @ BtPA

