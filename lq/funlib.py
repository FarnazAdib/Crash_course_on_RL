import numpy as np
import scipy.linalg as LA


class ADAM:
    def __init__(self, m, n, step_size=0.1, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
        self.m = m
        self.n = n
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.it_index = 0
        self.adam_M = np.zeros((self.m, self.n))
        self.adam_V = np.zeros((self.m, self.n))

    def opt_onestep(self, g):
        '''
        This function calculates one iteration of adam optimization. It takes the gradient of the cost function with
        respect to parameter theta and return dtheta. Note that you should use +dtheta when you are maximizing and
        -dtheta when minimizing.
        return the changes for the learning parameter
        :param g: The gradient of the loss function with respect to the parameter theta
        :return: dtheta by ADAM
        '''
        self.adam_M = self.beta1 * self.adam_M + (1 - self.beta1) * g
        self.adam_V = self.beta2 * self.adam_V + (1 - self.beta2) * (g * g)
        effective_step_size = self.step_size * np.sqrt(1 - self.beta2 ** (self.it_index + 1)) / (1 - self.beta1 ** (self.it_index + 1))
        self.it_index = self.it_index + 1
        return effective_step_size * self.adam_M / (np.sqrt(self.adam_V) + self.epsilon)


def inst_variable(x, y, z):
    '''
    Instrumental variable method
    Args:
        x: the input matrix [T n]
        y: the output matrix [T]
        z: the instrument [T n]
    Returns:
    the estimation of theta in y = x theta + n by instrumental variable
    '''
    T, n = x.shape
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    epsI = 0.00001 * np.eye(n)
    for t in range(T):
        A += np.outer(z[t, :], x[t, :]) / T
        B += np.dot(z[t, :], y[t] / T).reshape((n, 1))
    return LA.inv(A + epsI) @ B


def GtoP(G, K):
    '''
    :param G: The kernel of Q function
    :param K: The gain
    :return: The P associated with G and K
    '''
    _, n = K.shape
    M = np.concatenate((np.eye(n), K.T), axis=1)
    return M @ G @ M.T


def vecv(x):
    '''
    :param x: input vector of shape [T , n]
    :return: vector of x^2 of shape [T, n(n+1)/2]
    '''
    T, n = x.shape
    N = int(n * (n + 1) / 2)
    y = np.zeros((T, N))
    for t in range(T):
        yt = []
        for i in range(n):
            for j in range(i, n):
                if j == i:
                    yt.append(x[t, i] ** 2)
                else:
                    yt.append(2 * x[t, i] * x[t, j])
        y[t, :] = yt
    return y


def SquareMat(v, n):
    '''
    :param v: a vector
    :param n: dimension of the symmetric square matrix
    :return: a symmetric square matrix using v
    '''
    P = np.zeros((n, n))
    s = 0
    for i in range(n):
        e = s + n - i
        m = v[s:e].T
        P[i, i:] = m
        P[i:, i] = m
        s = e
    return P


def linear_func(p, x):
    return np.dot(p, x)