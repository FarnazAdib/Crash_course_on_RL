import numpy as np
import copy
from lq.dynamics import Linear_Quadratic
from lq.policies import LinK
from lq.funlib import ADAM


class PGRL:
    def __init__(self, sysdyn:Linear_Quadratic):
        # self.rand_seed = 1
        # np.random.seed(self.rand_seed)
        self.dyn = sysdyn
        self.n, self.m = self.dyn.B.shape

    def safeK(self, K, safeguard):
        if np.isnan(K).any():
            K = safeguard * np.ones((self.m, self.n))
        return K

    def random_search_linpolicy(self, K0, N, batch_size, T, explore_mag=0.04, step_size=0.05, safeguard=10):
        '''
        Random search.
        Gian initialization is not important
        :param N: Number of rollouts
        :param T: Time horizon
        :param explore_mag: Magnitude of the noise to explore
        :param step_size: the step size for learning
        :param batch_size: Number of directions per minibatches
        :param safeguard: The maximum absolute value of entries of the controller gain
        :return: Controller K by random search on LQ problem
        '''
        Lin_gain = LinK(K0)
        for k in range(N):
            reward_store = []
            batch = np.zeros((self.m, self.n))
            for j in range(batch_size):
                gain_randomness = np.random.randn(self.m, self.n)
                for sign in [-1, 1]:
                    K_test = LinK(Lin_gain.K + sign * explore_mag * gain_randomness)
                    _, _, costs, _ = self.dyn.one_rollout(K_test.lin_policy, T)
                    reward = - np.sum(costs) / T
                    batch += (reward * sign) * gain_randomness / 2 * explore_mag
                    reward_store.append(reward)

            dk = (step_size / np.std(reward_store) / batch_size) * batch
            Lin_gain.K += dk
            Lin_gain.K = np.minimum(np.maximum(Lin_gain.K, -safeguard), safeguard)
        return self.safeK(Lin_gain.K, safeguard)

    def uniform_random_search_linpolicy(self, K0, N, T, linf_norm=3):
        '''
        Uniform random search
        Initialization is not important
        :param N: Number of rollouts
        :param T: Time horizon
        :param linf_norm: Maximum controller value
        :return: Controller by uniform random linear policy
        '''

        Lin_gain = LinK(K0)
        best_K = Lin_gain.K
        best_reward = -float("inf")
        for k in range(N):
            Lin_gain.K = Lin_gain.uniform_sample_gain(linf_norm)
            _, _, costs, _ = self.dyn.one_rollout(Lin_gain.lin_policy, T)
            reward = - np.sum(costs)/T
            if reward > best_reward:
                best_reward = reward
                best_K = Lin_gain.K
        return best_K

    def pg_linpolicy(self, K0, N, batch_size, T, explore_mag=0.1,
                     step_size=0.1, beta1=0.9, beta2=0.999, epsilon=1.0e-8, safeguard=10):
        '''
        The policy gradient algorithm where the policy is considered to be linear. We use ADAM optimization.
        :param K0: The initial controller gain
        :param N: Number of iterations
        :param batch_size: number of batches per each step of optimization
        :param T: Trajectory length
        :param explore_mag: Magnitude of the noise to explore
        :param step_size: learning rate for Adam
        :param beta1: Adam related parameter
        :param beta2: Adam related parameter
        :param epsilon: Adam related parameter
        :param safeguard: The maximum absolute value of entries of the controller gain
        :return: The controller gain K
        '''

        # Initialize the controller
        Lin_gain = LinK(copy.copy(K0))
        Lin_gain.make_sampling_on(explore_mag)

        # A heuristic baseline to decrease the variance
        baseline = 0.0

        # Initializing Adam optimizer
        adam = ADAM(self.m, self.n, step_size=step_size, beta1=beta1, beta2=beta2, epsilon=epsilon)

        # start iteration
        for k in range(N):
            batch = np.zeros((self.m, self.n))
            reward = np.zeros(batch_size)

            # In each iteration, collect batches
            for j in range(batch_size):

                # Do one rollout
                states, actions, costs, _ = self.dyn.one_rollout(Lin_gain.sample_lin_policy, T)

                # Building the gradient of the loss with respect to gain
                actions_randomness = actions - Lin_gain.lin_policy(states)
                reward[j] = -np.sum(costs)/T
                batch += explore_mag**(-2) * ((reward[j] - baseline) / batch_size) * actions_randomness.T @ states

            # Update the baseline when batches are collected
            baseline = np.mean(reward)

            # Update the policy using ADAM
            dK = adam.opt_onestep(batch)
            Lin_gain.K += dK
        return self.safeK(Lin_gain.K, safeguard)

    def pg_vanilla_linpolicy(self, K0, N, batch_size, T, explore_mag=0.1,
                             step_size=0.1, beta1=0.9, beta2=0.999, epsilon=1.0e-8, safeguard=10):
        '''
        The vanilla policy gradient algorithm where the policy is considered to be linear. We use ADAM optimization.
        Different from the policy gradient, in each step rewards to go for that step is used.
        :param N: Number of rollouts
        :param batch_size: Number of stochastic gradient per minibatch
        :param T: Trajectory length
        :param explore_mag: Magnitude of the noise to explore
        :param step_size: The step size for ADAM optimization
        :param beta1: Beta1 in ADAM Optimization
        :param beta2: Beta2 in ADAM optimization
        :param epsilon: Epsilon in ADAM optimization
        :param safeguard: The maximum absolute value of entries of the controller gain
        :return: Controller gain by vanilla PG using ADAM
        '''
        # Initialize the controller
        Lin_gain = LinK(copy.copy(K0))

        # A heuristic baseline to decrease the variance
        Lin_gain.make_sampling_on(explore_mag)
        baseline = 0.0

        # Initializing Adam optimizer
        adam = ADAM(self.m, self.n, step_size=step_size, beta1=beta1, beta2=beta2, epsilon=epsilon)

        # start iteration
        for k in range(N):
            batch = np.zeros((self.m, self.n))
            reward = np.zeros(batch_size)

            # In each iteration, collect batches
            for j in range(batch_size):

                # Do one rollout
                states, actions, costs, _ = self.dyn.one_rollout(Lin_gain.sample_lin_policy, T)

                # Building the gradient of the loss with respect to gain
                actions_randomness = actions - Lin_gain.lin_policy(states)
                reward[j] = -np.sum(costs) / T
                for t in range(T):
                    reward_to_go = -np.sum(costs[t:]) / T
                    batch += (explore_mag ** (-2) * (reward_to_go - baseline) / batch_size) * \
                             np.outer(actions_randomness[t, :], states[t, :])

            # Update the baseline after collecting batches
            baseline = np.mean(reward)

            # Update the policy using ADAM after collecting batches in each iteration
            dK = adam.opt_onestep(batch)
            Lin_gain.K += dK
        return self.safeK(Lin_gain.K, safeguard)
