import tensorflow as tf

import numpy as np
from functools import reduce

import sys
sys.path.insert(0, '/home/nagy729krisztina/M4_treedom')
import config as cfg

class Vgg19:

	######################################################
	# Load variables form npy to build the VGG
	#  Args:
	#    vgg19_npy_path: path to the npy file storing the trained parameters
	#    trainable: boolean, if True, the paramaters can be trained and dropout will be turned on
	#	 dropout: chance of dropout for fully connected layers
	#  Returns:
	#    None
	######################################################
    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

	######################################################
	# Load variables form npy to build the VGG
	#  Args:
	#    rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
	#    train_mode: a boolean, if True, dropout will be turned on
	#  Returns:
	#    self.output: the output of the network, the result of classification
	######################################################
    def inference_graph(self, grayscale, train_mode=None, image_pixels = cfg.image['height']*cfg.image['width'], num_classes = 2):

        grayscale_scaled = grayscale * 255.0
        
        ### Dimension info: images_reshaped shape: expected to be [batch_size, 64, 64, 1]
        grayscale_reshaped = tf.reshape(grayscale_scaled, [-1,cfg.image['height'],cfg.image['width'],1], name="grayscale_reshaped" )
        print("grayscale_reshaped size:")
        print(grayscale_reshaped.get_shape())
        
        ### VGG Net definition ###
        self.conv1 = self.conv_layer(grayscale_reshaped, 5, 1, 64, "conv1")
        self.pool1 = self.max_pool(self.conv1, 'pool1')

        self.conv2 = self.conv_layer(self.pool1, 5, 64, 128, "conv2")
        self.pool2 = self.max_pool(self.conv2, 'pool2')

        self.conv3 = self.conv_layer(self.pool2, 9, 128, 256, "conv3")
        self.pool3 = self.max_pool(self.conv3, 'pool3')

        self.conv4 = self.conv_layer(self.pool3, 9, 256, 512, "conv4")
        self.pool4 = self.max_pool(self.conv4, 'pool4')

        self.conv5 = self.conv_layer(self.pool4, 9, 512, 512, "conv5")
        self.pool5 = self.max_pool(self.conv5, 'pool5')
        
        # 2048 = ((64 // (2 ** 5)) ** 2) * 512
        self.fc6 = self.fc_layer(self.pool5, 2048, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        # if train_mode is not None:
            # self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        # elif self.trainable:
            # self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        
        # if train_mode:
            # self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        # if train_mode is not None:
            # self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        # elif self.trainable:
            # self.relu7 = tf.nn.dropout(self.relu7, self.dropout)
            
        # if train_mode:
            # self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, "fc8")
		# name cannot be "fc8", because the program would try to load the weights to 1000 classes.
		# output nodes number modified: 1000 -> 2
		# we only have 2 classes, the layer is thus renamed to 'fc8_modified'
        
        self.output = tf.nn.softmax(self.fc8, name="prob")
        #self.truncate1 = tf.maximum(self.fc8, -1, name="truncate1")
        #self.output = tf.minimum(self.truncate1, 1, name="output")
        
        #self.data_dict = None
        
        return self.output
		
	######################################################
	# Helper functions for defining layers
	######################################################
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, filter, in_channels, out_channels, name):
        with tf.variable_scope(name):
            
            weights = tf.Variable(tf.truncated_normal([filter, filter, in_channels, out_channels], mean=0.0, stddev=0.01))
            biases = tf.Variable(tf.truncated_normal([out_channels], mean=0.0, stddev=0.01))

            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            
            weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0.0, stddev=0.01))
            biases = tf.Variable(tf.truncated_normal([out_size], mean=0.0, stddev=0.01))

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc