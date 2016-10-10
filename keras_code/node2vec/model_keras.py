
# coding: utf-8

# In[31]:

#! /usr/bin python


# In[32]:

from __future__ import print_function
import numpy as np
import pdb

# In[33]:

from keras import backend as K


# In[34]:

from keras.models import Model
from keras.layers import Lambda, Input, Flatten, Dense
from keras.layers.core import Reshape, Permute
from keras.engine import Layer
from keras import initializations
from keras.layers.embeddings import Embedding
from keras.models import Sequential


# In[35]:

from preprocess import get_size, gen_batch, parse_test
from sklearn.metrics import accuracy_score

# In[9]:

class NodeEmbedLayer(Layer):
    def __init__(self,hidden_dim, **kwargs):
        self.hidden_dim = hidden_dim
        self.init = initializations.get('glorot_uniform')
        super(NodeEmbedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        embed_dim = input_shape[1]
        self.W_p = self.init((embed_dim, self.hidden_dim))
        self.W_m = self.init((embed_dim, self.hidden_dim))
        self.b = K.zeros((self.hidden_dim, ))

    def call(self, x, mask=None):
        X = x[:, :, 0] * x[:, :, 1]
        Y = K.abs(x[:, :, 0] - x[:, :, 1])
        z = K.dot(X, self.W_p) + K.dot(Y, self.W_m)
        return K.tanh(z) #+ self.b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.hidden_dim)


def model_node2vec(embed_dim=64):
    N = get_size()
    model = Sequential()
    model.add(Embedding(N+1, embed_dim, input_length=2))
    model.add(Permute((2,1)))
    model.add(NodeEmbedLayer(15))
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[11]:

if __name__ == "__main__":
    n_epochs = 25
    model = model_node2vec()
    model.compile('adam', 'binary_crossentropy')
    X_test, y_test = parse_test()
    for epoch in xrange(n_epochs):
        for batch in gen_batch(64):
            #pdb.set_trace()
            x = model.train_on_batch(batch[0], batch[1])
            print(x, end="\r")
        y_pred = model.predict_classes(X_test)
        score = accuracy_score(y_test, y_pred)
        print("\n Acc on test set is %f"%score)
    pdb.set_trace()


# In[ ]:



