# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:12:14 2017

@author: enggsudarshan
"""

import tflearn #TFlearn library
from tflearn.layers.conv import conv_2d,max_pool_2d #TFlearn library for Convolutional and Pooilng Layers
from tflearn.layers.core import input_data, dropout, fully_connected #TFlearn library for Dropout, Input Data and Fully Connected Layer
from tflearn.layers.estimator import regression #TFlearn for regression Layer
import tflearn.datasets.mnist as mnist #TFlearn for MNIST dataset

X, Y, test_x, test_y = mnist.load_data(one_hot = True)
#one_hot is used for 1. Multiclass classification

"""
We have 10 classes 0-9
0 = 0
1 = 1
..
..
.

basically one_hot makes 0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
...
...
...

"""

X = X.reshape([-1,28,28,1]) #reshaping 784 pixel image into flat 28*28
test_x = test_x.reshape([-1,28,28,1]) #reshaping 784 pixel image into flat 28*28

convnet = input_data(shape=[None,28,28,1], name = 'input')

convnet = conv_2d(convnet, 32, 2, activation='relu') #Convolutional Layer 1
convnet = max_pool_2d(convnet,2)    #Pooling Layer 1

convnet = conv_2d(convnet, 64, 2, activation='relu') #Convolutional Layer 2
convnet = max_pool_2d(convnet,2)    #Pooling Layer 2

convnet = fully_connected(convnet, 1024, activation='relu') #Fully Connected Layer
convnet = dropout(convnet, 0.8) # basically 80% of neurons will fire

convnet = fully_connected(convnet, 10, activation = 'softmax') #Output Layer with Softmax
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets') #Regression layer

model = tflearn.models.DNN(convnet, tensorboard_dir='log') #Deep Neural Network Model with TensorBoard log directory

#After training commenting this part of code

model.fit({'input':X},{'targets':Y}, n_epoch=10, 
          validation_set=({'input':test_x},{'targets':test_y}),
          snapshot_step=500, show_metric=True, run_id='mnist')
          
model.save('tflearncnn.model') #saves your weights.


#model.load('tflearncnn.model')
print(model.predict([test_x[1]])) #Print Sample Model to check for Accuracy
