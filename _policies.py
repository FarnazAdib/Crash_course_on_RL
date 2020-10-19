# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 08:49:41 2019

@author: farad59
"""

# Support code for cem.py

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0) # if y<0, then it returns 1 otherwise it returns 0
        return a

class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, n_in, n_out):
        assert len(theta) == (n_in + 1) * n_out
        self.W = theta[0 : n_in * n_out].reshape(n_in, n_out)
        self.b = theta[n_in * n_out : None].reshape(1, n_out)
    def act(self, ob):
        a = ob.dot(self.W) + self.b
        return a