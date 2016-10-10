#! /usr/bin python

import numpy as np
import math

DATA_DIR = '../data/'


# Parser for network info. presented as edge-list
def parse_authornet(file_name):
    author_path = DATA_DIR + file_name
    f = open(author_path)
    # Array to store network info.
    net = []
    y = []
    for line in f:
        src_node, tar_node, label = map(lambda x: int(x), line.rstrip().split())
        net.append(list([src_node, tar_node]))
        y.append(label)
    y = np.array(y)
    return np.array(net), y


def get_size():
    X, y = parse_authornet('train.txt')
    n = max(np.max(X[:, 0]), np.max(X[:, 1]))
    return n


def parse_test():
    X_test, y_test = parse_authornet('test1.txt')
    return X_test, y_test

# Generates batch for training
def gen_batch(batch_size):
    # Get the data matrix from the parser
    X, y = parse_authornet('train.txt')
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in xrange(0, n_batches):
        if i < n_batches - 1:
            batch = X[i*batch_size:(i+1) * batch_size, :]
            Y = y[i*batch_size:(i+1) * batch_size]
            yield batch, Y

        else:
            batch = X[end:, :]
            n += X[end:, :].shape[0]
            Y =  y[end:]
            yield batch, Y


if __name__ == "__main__":
    for batch in gen_batch(64):
        print batch[0].shape, batch[1].shape
