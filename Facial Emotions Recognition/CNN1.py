# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:35:55 2017

@author: enggsudarshan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:51:59 2017

@author: enggsudarshan
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def error_rate(targets, predictions):
    return np.mean(targets != predictions)
    
def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
            #print(X)
    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    #print(X)
    X = X.reshape(N, 1, d, d)
    #print(X)
    #print(N,D,d)
    return X, Y


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ConvPoolLayer(object):
                    # mi--> #input feature maps
                    # mo--> #output feature maps            
                    # fw--> filter width
                    # fw--> filter height
                    # poolsz --> pool size
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2,2)):
        sz = (fw, fh, mi, mo)
        W0 = init_filter(sz,poolsz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32) #For bias
        self.b = tf.Variable(b0)
        self.poolsz = poolsz#Keep track of pool size
        self.params = [self.W,self.b]   # Keep track of params
    
    # Define Forward action for ConvPool Layer
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1,1,1,1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out,self.b)
        
        pool_out = tf.nn.max_pool(conv_out,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        return tf.tanh(pool_out) #Activation Function as Tanh
        
        

class CNN(object):
    def __init__(self,convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
    
        #lr = learning rate
        #reg = regularization 
        #eps = epsilon
    def fit(self, X, Y, lr= 1e-3, mu=0.99, reg = 10e-4, decay=0.99999, eps = 10e-3, batch_sz=30,epochs=3, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        K = len(set(Y)) #Unique values in Y
        
        #Creating Validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:] #First 1000
        X, Y = X[:-1000], Y[:-1000] #Set X, Y to remaining
        Yvalid_flat = np.argmax(Yvalid, axis=1) #For error claculation
        
        #Initialize ConvPool layer
        N, d, d, c = X.shape
        mi = c  #Input feature map = color
        outw = d
        outh = d
        self.convpool_layers = []   #Save convool layers in a list
        print('convpool_layer_sizes',self.convpool_layer_sizes)
        
        for mo, fw, fh in self.convpool_layer_sizes:
            layers = ConvPoolLayer(mi, mo, fw, fh)  #Initialize layers
            self.convpool_layers.append(layers)
            outw = outw / 2 #Divide by 2 because of pooling layer
            outh = outh / 2
            mi = mo
            
        #Initialize Hidden layers
        self.hidden_layers  = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh
        count=0 #As these Id's will be passed into hidden layers
        print('hidden_layer_sizes',self.hidden_layer_sizes)
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count = count + 1
            
        #Initialize Logistic Regression
        W, b = init_weight_and_bias(M1,K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b,'b_logreg')
            
        self.params = [self.W, self.b]
        for h in self.convpool_layers:
            self.params = self.params + h.params
            
        for h in self.hidden_layers:
            self.params = self.params + h.params
            
        #Define TensorFlow functions and Variables
        tfX = tf.placeholder(tf.float32, shape=(None, d, d, c))
        tfY = tf.placeholder(tf.float32, shape=(None, K))
        act = self.forward(tfX)
        
        #Calculate Regularization Cost
        
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        
        #Calculate Final Cost
                                                                    #Activation, Indicator Martix of targets
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=tfY)) + rcost
        
        #Calculate prediction
        
        prediction = self.predict(tfX)
        
        #Define train function
        
       # train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        #Calculate No of batches
        n_batches = N/batch_sz
        
        #Initialize cost array
        costs=[]
        
        #Initialize all variables
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)
            
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                    
                    session.run(train_op, feed_dict={tfX:Xbatch, tfY:Ybatch})
                    
                    if j%20 == 0:
                        c = session.run(cost, feed_dict={tfX:Xvalid, tfY:Yvalid})
                        costs.append(c)
                        
                        #Calculate prediction
                        p = session.run(prediction, feed_dict={tfX:Xvalid, tfY:Yvalid})
                        
                        #Calculate error rate
                        e = error_rate(Yvalid_flat, p)
                        print('i',i,'j',j,'n_batches',n_batches,'cost',c, 'error_rate',e)
                        
        if show_fig:
            plt.plot(costs)
            plt.show()
    
    def forward(self, X):
        Z = X#Z-looping variable
        
        #Loop through ConvPool layers
        for c in self.convpool_layers:
            Z = c.forward(Z)
            
        #Flatten Z
        Z_shape = Z.get_shape().as_list()
        
        #Reshape Z
        Z = tf.reshape(Z, [-1,np.prod(Z_shape[1:])])    #-1 As we dont know the shape
        for h in  self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b  #Return Activation and not the Softmax
        
    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)
                        
def main():
    X,Y = getImageData()
    
    X = X.transpose((0,2,3,1))  #0=N, 1-Color, 2-Width, 3-Height
                            #No of feature maps, feature width, feature height
                            # No of convpool layers=2
    #print(X)
    model = CNN(convpool_layer_sizes=[(20,5,5),(20,5,5)], hidden_layer_sizes = [500,300],)
    model.fit(X,Y,show_fig=True)
    

if __name__ == '__main__':
    main()