import datetime   #datetime library
import tensorflow as tf  #tensorflow library
from tensorflow.examples.tutorials.mnist import input_data #library to fetch MNIST data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True) #one_hot is used for 1. Multiclass classification

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

#total 10 classes
n_classes = 10
#batch_size = 100    #100 images at a time
batch_size = 128    #128 images at a time

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

#matrix is height * width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')


def conv_2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

# stride means move ... in our case move 1 pixel at a time
    
def max_pool2d(x):
                            #size of window               movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# 2x2 pooling..

##Convolutional Neural Network Model
def convolutional_neural_network_model(x):
    
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),	# 5x5 convolution, Depth =1, #features = 32. Will produce a 5x5x32 output.
	       'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),    # 5x5 convolution, Depth =32, #features = 64. Will produce a 5x5x64 output.
	       'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),   #We started with 28*28, then after max pool1-> 14*14, max pool2-> 7*7, with 1024 nodes
	       'out':tf.Variable(tf.random_normal([1024, n_classes]))  #Output layer with 1024 neurons, 10 classes
                      }

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),		# 32 bias nodes	
	       'b_conv2':tf.Variable(tf.random_normal([64])),  		# 64 bias nodes	        
	       'b_fc':tf.Variable(tf.random_normal([1024])),		     #1024 bias nodes	
	       'out':tf.Variable(tf.random_normal([n_classes]))}        #10 bias nodes
    
    x = tf.reshape(x, shape=[-1,28,28,1])   #reshaping 784 pixel image into flat 28*28
    
    conv1 = conv_2d(x,weights['W_conv1'])   #1st Convolution Layer
    conv1 = max_pool2d(conv1)               #1st Pooling Layer

    conv2 = conv_2d(conv1,weights['W_conv2']) #2nd Convolutional Layer
    conv2 = max_pool2d(conv2)                   #2nd Pooling Layer

    fc = tf.reshape(conv2,[-1,7*7*64])      #For using Classical Neural Network, reshape the tensor back to its original form
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])    #Activation Function: ReLU
    
    fc = tf.nn.dropout(fc, keep_rate) # basically 80% of neurons will fire

    output = tf.matmul(fc,weights['out']) + biases['out']    #Output
 
    return output
    


def train_neural_network(x):
    prediction = convolutional_neural_network_model(x)
    #Calculate the difference of the prediction we got and the known label
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    
    # Does have a parameter learning rate. by default 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #cycles: feef forward + backprop
    how_many_epochs = 10
    t0 = datetime.datetime.now()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
# Training the network
        for epochs in range(how_many_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):        # _ is for variable we don't care about
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            t1 = datetime.datetime.now()
            print('Epoch',epochs, 'completed out of', how_many_epochs, 'loss:', epoch_loss) #After every epoch, print the current epoch with the time taken for execution.
            print('Time taken',t1-t0)

# Testing the data            
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))        
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
	for _ in range(int(mnist.test.num_examples/batch_size)):
		test_x,test_y = mnist.test.next_batch(batch_size)
        	#print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))        #Print accuracy for each epoch.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
train_neural_network(x)

