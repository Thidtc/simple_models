#!/usr/bin/python
# coding=utf-8

import numpy as np
from utils import sigmoid

class RBM(object):
    def __init__(self, n_v, n_h, init_param=None, lr=0.1, k=1):
        self.n_v = n_v
        self.n_h = n_h
        if init_param is not None:
            self.W, self.a, self.b = init_param
            assert self.W.shape == (n_v, n_h)
            assert self.a.shape == n_v
            assert self.b.shape == n_h
        else:
            self.W = np.random.uniform(low=-1.0/n_v, high=1.0/n_v, size=(n_v, n_h))
            self.a = np.zeros((n_v, ))
            self.b = np.zeros((n_h, ))
        
        self.lr = lr
        self.k = k
    
    def fit(self, X, n_epochs=5, verbose=True):
        '''
        Gradient decrease
        '''
        for epoch in range(n_epochs):
            d_W, d_a, d_b = self.CD(X)
            self.W += self.lr * d_W
            self.a += self.lr * d_a
            self.b += self.lr * d_b
            
            if not verbose:
                if epoch % 100 == 0:
                    error = self.reconstruct_cross_entory(X)
                    print('Epoch {epoch}, error {error}'.format(epoch=epoch, error=error))
    
    def CD(self, X):
        '''
        Contrastive divergence procedure
        '''
        # Variable with 'value' means the continues cell value, that is, the
        # probability of cell being activated, variable with 'sample' means
        # the discrete cell value, that is, the sampled value of cell
        h_value_0, h_sample_0 = self.sample_h_given_v(X)
        v_sample = X
        for step in xrange(self.k):
            v_value, v_sample, h_value, h_sample =\
                self.gibbs_sample(v_sample)
        
        d_W = np.dot(X.T, h_value_0) - np.dot(v_sample.T, h_value)
        d_a = np.mean(X - v_sample, axis=0)
        d_b = np.mean(h_value_0 - h_value, axis=0)

        return d_W, d_a, d_b

    
    def gibbs_sample(self, v):
        '''
        Sample new visibale and hidden layer value
        '''
        h_value, h_sample = self.sample_h_given_v(v)
        v_value, v_sample = self.sample_v_given_h(h_sample)

        return v_value, v_sample, h_value, h_sample

    def sample_h_given_v(self, v):
        '''
        Sample hidden layer from the visible layer
        '''
        value = self.propup(v)
        sample = np.random.binomial(size=value.shape, n=1, p=value)
        return (value, sample)

    def sample_v_given_h(self, h):
        '''
        Sample visible layer from the hidden layer
        '''
        value = self.propdown(h)
        sample = np.random.binomial(size=value.shape, n=1, p=value)
        return (value, sample)

    def propup(self, v):
        '''
        Calculate hidden layer value based on visible layer value
        visible layer -> hidden layer
        '''
        return sigmoid(np.dot(v, self.W) + self.b)

    def propdown(self, h):
        '''
        Calculate visible layer value based on hidden layer value
        hidden layer -> visible layer
        '''
        return sigmoid(np.dot(h, self.W.T) + self.a)

    def reconstruct(self, v):
        '''
        Reconstruct visible layer
        '''
        h = sigmoid(np.dot(v, self.W) + self.b)
        reconstruct_v = sigmoid(np.dot(h, self.W.T) + self.a)
        return reconstruct_v

    def reconstruct_cross_entory(self, X):
        reconstruct_X = self.reconstruct(X)
        error = -np.mean(np.sum(X * np.log(reconstruct_X) + (1 - X) * np.log(1 - reconstruct_X), axis=1))
        return error


def TEST_RBM(lr = 0.1, k = 1, n_epochs=1000):
    data = np.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])

    rbm = RBM(n_v=6, n_h=2, lr=lr, k=k)
    rbm.fit(data, n_epochs=n_epochs)

    v = np.array([[1, 1, 1, 0, 0, 0],\
                 [0, 0, 0, 1, 1, 0]])
    reconstruct_v = rbm.reconstruct(v)
    print(reconstruct_v)

if __name__ == '__main__':
    TEST_RBM()