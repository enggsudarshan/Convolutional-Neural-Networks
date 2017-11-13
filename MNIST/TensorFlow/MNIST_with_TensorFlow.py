"""
inputs > weights > hidden layer 1 (activation function) > weights > 
hidden layer 2 (activation function) > weights > output layer

compare output to actual output > using cost function(cross entropy) / loss function

Use optimization function to minimize cost(Adam optimizer, SGD ....)

This is done using backpropogation

Feed forward  + Backproc = epoch!! i.e one cycle


"""
import os
import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data", one_hot=True) #one_hot is used for 1. Multiclass classification

log_path = '/home/enggsudarshan/Desktop/MNIST/TensorFlow/log'

from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = None



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
"""
# 3 hidden layers each with 500 nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
"""

#total 10 classes
n_classes = 10
#batch_size = 100    #100 images at a time
batch_size = 128    #128 images at a time

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#matrix is height * width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

"""
with tf.device("/cpu:0"):
    embedding_var = tf.Variable(tf.stack(mnist.test.images, axis=0), trainable=False, name='embedding')
"""

def conv_2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

# stride means move ... in our case move 1 pixel at a time
    
def max_pool2d(x):
                            #size of window               movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 2x2 pooling..

##Convolutional Neural Network Model
def convolutional_neural_network_model(x):
    
    # input_data*weights + biases
    """
    Why do we need a bias?
    
    If all of the input data is 0... then no neuron will ever fire.
    Also can make your network more dynamic
    """
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),			# 5x5 convolution, it will take 1 input and produce 32 features(output)
	       'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
	       'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
	       'out':tf.Variable(tf.random_normal([1024, n_classes]))
                      }

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),			
	       'b_conv2':tf.Variable(tf.random_normal([64])),
	       'b_fc':tf.Variable(tf.random_normal([1024])),
	       'out':tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.reshape(x, shape=[-1,28,28,1])
    
    with tf.name_scope('Convolutional1'):
        conv1 = conv_2d(x,weights['W_conv1']+biases['b_conv1'])
        conv1 = tf.nn.relu(conv1)
    with tf.name_scope('MaxPool1'):
        conv1 = max_pool2d(conv1)
    
    with tf.name_scope('Convolutional2'):
        conv2 = conv_2d(conv1,weights['W_conv2']+biases['b_conv2'])
        conv2 = tf.nn.relu(conv2)
    with tf.name_scope('MaxPool2'):
        conv2 = max_pool2d(conv2)    

    with tf.name_scope('FullyConnected'):
        fc = tf.reshape(conv2,[-1,7*7*64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    
    with tf.name_scope('Dropout'):
        fc = tf.nn.dropout(fc, keep_rate) # basically 80% of neurons will be dropped
    with tf.name_scope('Output'):
        output = tf.matmul(fc,weights['out']) + biases['out']    
 
    return output
    


def train_neural_network(x):
    prediction = convolutional_neural_network_model(x)
    #Calculate the difference of the prediction we got and the known label
    with tf.name_scope('Loss'):  
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    
    # Does have a parameter learning rate. by default 0.001
    with tf.name_scope('AdamOptimizer'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #cycles: feef forward + backprop
    with tf.name_scope('Accuracy'):
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    
    init = tf.global_variables_initializer()
    how_many_epochs = 10
    t0 = datetime.datetime.now()
    tf.summary.scalar("loss",cost)
    tf.summary.scalar("accuracy",accuracy)                 
    
    merged_summary_op = tf.summary.merge_all()   
    
    metadata = os.path.join(log_path, 'metadata.tsv')    
    
    with open(metadata, 'w') as metadata_file:
        for row in range(10000):
            c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
            metadata_file.write('{}\n'.format(c))


    with tf.Session() as sess:
        sess.run(init)
        

        summary_writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())

# Training the network
        for epochs in range(how_many_epochs):
            epoch_loss = 0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):        # _ is for variable we don't care about
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _, c, summary = sess.run([optimizer,cost,merged_summary_op], feed_dict = {x:epoch_x, y:epoch_y})
                summary_writer.add_summary(summary,epochs*total_batch + i)
                
                epoch_loss += c / total_batch
            t1 = datetime.datetime.now()
            print('Epoch',epochs, 'completed out of', how_many_epochs, 'loss:', epoch_loss)
            print('Time taken',t1-t0)
            
            
            """
            saver = tf.train.Saver()
            config = projector.ProjectorConfig()
            embed = config.embeddings.add()
            embed.tensor_name = embedding_var.name
            embed.metadata_path = os.path.join(log_path, 'metadata.tsv')
            projector.visualize_embeddings(summary_writer, config)
            saver.save(sess, os.path.join(log_path, "model.ckpt"), global_step=how_many_epochs)
            """
# Testing the data 
        
        for _ in range(int(mnist.test.num_examples/batch_size)):
            test_x,test_y = mnist.test.next_batch(batch_size)
            #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
            
        images = tf.Variable(mnist.test.images, name='images')
        saver = tf.train.Saver([images])
        sess.run(images.initializer)
        saver.save(sess, os.path.join(log_path, 'images.ckpt'))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(tf.summary.FileWriter(log_path), config)
        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
train_neural_network(x)