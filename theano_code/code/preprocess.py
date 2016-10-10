import numpy as np
import io
import pdb
from collections import defaultdict
import random

#path = '../data/release-youtube-links.txt'
path = '../data/p2p-Gnutella08.txt'
u = []
v = []
l = []

edge = defaultdict(list)
data = []

#@profile
#def read_graph():
for line in io.open(path).readlines():
    U,V = line.split()
    u.append(int(U))
    v.append(int(V))
    l.append(1)

u0 = u[0]
edge[u0].append(v[0])
for i in xrange(1,len(u)):
    if u[i] == u0:
        edge[u[i]].append(v[i])
    else:
        u0 = u[i]
        edge[u0].append(v[i])

V_set = set(v)

#print "edge list created"
#for i in xrange(len(u)):
#    V = v[i]
#    #pdb.set_trace()
#    rand = random.sample((V_set - set([V])), 1)
#    u.append(u[i])
#    v.append(V)
#    l.append(0)


#@profile
#def gen_neg_samples():
for U in edge.keys():
    pos_edges = edge[U]
    pos_edges = set(pos_edges)
    neg_samples = V_set - pos_edges
    neg_edges = random.sample(neg_samples, len(pos_edges))
    temp  = [U] * len(pos_edges)
    u.extend(temp)
    v.extend(neg_edges)
    l.extend([0] * len(pos_edges))

u = np.array(u)
v = np.array(v)
l = np.array(l)
data = np.zeros((len(u),3),dtype=np.int32)
data[:, 0] = u[:]
data[:, 1] = v[:]
data[:, 2] = l[:]

np.random.shuffle(data)
#pdb.set_trace()

#if __name__ == "__main__":
#    read_graph()
#    gen_neg_samples()
