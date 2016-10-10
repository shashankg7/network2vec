from theano import tensor as T
import theano
import numpy as np
from preprocess import data

class node2vec(object):
    def __init__(self, n, d, h):
        self.n = n
        self.d = d
        self.h = h
        self.W = theano.shared(np.random.uniform(low = - np.sqrt(6.0/float(n + d)),\
                                   high =  np.sqrt(6.0/float(n + d)),\
                                   size=(n,d)).astype(theano.config.floatX))
        self.W1 = self.W
        self.Wm = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.Wp = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(h + d)),
                                    high = np.sqrt(6.0/float(h+d)),
                                    size=(h,d)).astype(theano.config.floatX))
        self.b1 = theano.shared(np.zeros(h, dtype=theano.config.floatX))
        self.b2 = theano.shared(np.zeros(2, dtype=theano.config.floatX))
        self.U = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
                                              high = np.sqrt(6.0/float(2 + h)),
                                              size=(2,h)).astype(theano.config.floatX))
        self.params = [self.Wm, self.Wp, self.b1, self.b2, self.U]

    def model(self, lr=0.001):
        # theano matrix storing node embeddings
        X = T.imatrix()
        # Target labels for input
        y = T.ivector()
        # Extract the word vectors corresponding to inputs
        U = self.W[X[:,0],:]
        V = self.W[X[:,1],:]
        hLm = U * V
        hLp = abs(U - V)
        hL = T.tanh(T.dot(hLm, self.Wm.T) + T.dot(hLp, self.Wp.T) + self.b1)
        params = [U , V]
        params.extend(self.params)
        # Likelihood
        l = T.nnet.softmax(T.dot(hL,self.U.T) + self.b2)[T.arange(y.shape[0]), y]
        cost = - T.log(l).sum()
        grads = T.grad(cost, params)
        #updates1 = [(self.W1, T.inc_subtensor(self.W[X[:, 0]], grads[0]))]
        #updates2 = [(self.W, T.inc_subtensor(self.W1[X[:, 1]], grads[1]))]
        self.W1 = T.set_subtensor(self.W1[X[:,0]], self.W1[X[:,0]] - lr * grads[0])
        self.W1 = T.set_subtensor(self.W1[X[:,1]], self.W1[X[:,1]] - lr * grads[1])
        updates1 = [(self.W, self.W1)]
        updates3 = [(param, param - lr * grad) for (param, grad) in zip(self.params, grads[2:])]
        updates = updates1  + updates3
        self.gd = theano.function([X,y], cost, updates=updates)



