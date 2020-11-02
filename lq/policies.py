import numpy as np

class LinK:
    def __init__(self, K):
        # self.rand_seed = 1
        # np.random.seed(self.rand_seed)
        self.K = K
        self.m, self.n = self.K.shape
        self.stddev = 0.0

    def lin_policy(self, x):
            '''
            A linear policy u=K x
            :param x: Input of shape   T, n
            :return: the policy of shape T, m
            '''
            return x @ self.K.T

    def make_sampling_on(self, stddev):
        self.stddev = stddev

    def sample_lin_policy(self, x):
        '''
        Sample the given policy
        :param x: Input of shape   T, d
        :param stddev: Standard deviation
        :return: action sampled from a gaussian distribution with mean x @ K.T and variance stddev
        '''
        return x @ self.K.T + self.stddev * np.random.randn(len(x), self.m)

    def uniform_sample_gain(self, l_max):
        '''
        Uniformly sample a linear gain
        Args:
            l_max: the maximum value for the absolute values of the entries
        Returns: A random gain
        '''
        return np.random.uniform(-l_max, l_max, (self.m, self.n))

