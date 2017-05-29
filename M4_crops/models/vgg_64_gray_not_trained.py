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
        #if load is False, trainable has to be True
        self.conv1_1 = self.conv_layer(grayscale_reshaped, 1, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        # 2048 = ((64 // (2 ** 5)) ** 2) * 512
        self.fc6 = self.fc_layer(self.pool5, 2048, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        
        if train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
            
        if train_mode:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, "fc8")
        
        self.output = tf.nn.softmax(self.fc8, name="prob")
        
        return self.output
		
	######################################################
	# Helper functions for defining layers
	######################################################
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            
            filter = 3
            
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