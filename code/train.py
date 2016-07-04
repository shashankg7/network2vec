from preprocess import data
from model import node2vec
import pdb

n = data.shape[0]
d = 10
h = 10
node2vec = node2vec(n,d,h)
node2vec.model()

def training(batch_size):
    for i in xrange(0, data.shape[0], batch_size):
        X = data[i:(i+batch_size), 0:2]
        y = data[i:(i+batch_size), 2]
        #pdb.set_trace()
        print node2vec.gd(X, y)

if __name__ == "__main__":
    training(32)
