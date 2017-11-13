# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:30:30 2017

@author: enggsudarshan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:27:25 2017

@author: enggsudarshan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:59:22 2017

@author: enggsudarshan
"""


import cv2     #resizing th image
import numpy as np   #for arrays
import os   #for directory access
from random import shuffle  #shuffle images
from tqdm import tqdm  #professional looping with progressbar


import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import tensorflow as tf
tf.reset_default_graph()

import matplotlib.pyplot as plt



TRAIN_DIR = '/home/enggsudarshan/Desktop/My Image Predictions/train'
TEST_DIR = '/home/enggsudarshan/Desktop/My Image Predictions/test'

IMG_SIZE = 50   #resize image 50x50
LR = 1e-3

MODEL_NAME = 'my-training-{}-{}.model'.format(LR, '6.1-layer-convpool-basic')   
#when you save the model you will be able to know what model it is later

#converting to one-hot array

def label_img(img):
    word_label = img.split('.')[-3] #eg: sud.25.png
    if word_label == 'Others':
        return [1,0]
    elif word_label == 'Sud':
        return [0,1]
        
def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)   #in future we can load this file
    return train_data
    

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0] #to identify whether th image is of a dog or cat
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img), img_num])
    np.save('test_data.npy', test_data)   #in future we can load this file
    return test_data

# Train data 
train_data = create_train_data()

#train_data = np.load('train_data.npy')

#CNN
#Input Layer
convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1], name = 'input')

#Layer1
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet,2)

#Layer2
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet,2)




'''
#Layer6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

'''
#Layer6
convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.8)


#Output Layer
convnet = fully_connected(convnet, 2, activation = 'softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


#print('Till Here...\n\n')

model = tflearn.models.DNN(convnet, tensorboard_dir='log')

#print('Till Here 2...\n\n')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!!')

train = train_data[:-10] # Training data will be all exculding last 10 images
test = train_data[-10:]    #For accuracy

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


#print(train_data)
#print(X)

#print('Till Here...3\n\n')

model.fit({'input':X},{'targets':Y}, n_epoch=30, 
          validation_set=({'input':test_x},{'targets':test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
          
#print('Till Here...4\n\n')

test_data = process_test_data()

#test_data = np.load('test_data.npy')


fig = plt.figure()

for num, data in enumerate(test_data[:25]):
    #cat [1:0]
    #dog [0:1]


    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(5,5,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1:
        str_label = 'Sudarshan'
    else:
        str_label = 'Not Sudarshan'
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

#model.save(MODEL_NAME)
