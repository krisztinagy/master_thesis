import tensorflow as tf

import numpy as np
from functools import reduce

import sys
sys.path.insert(0, '/home/nagy729krisztina/M4_treedom')
import config as cfg

VGG_MEAN = [103.939, 116.779, 123.68]


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
    def inference_graph(self, rgb, train_mode=None, image_pixels = cfg.image['height']*cfg.image['width'], num_classes = 2):
	
        rgb_scaled = rgb * 255.0
        
        ### Dimension info: images_reshaped shape: expected to be [100, 60, 80, 1]
        rgb_reshaped = tf.reshape(rgb_scaled, [-1,cfg.image['height'],cfg.image['width'],3], name="rgb_reshaped" )
        #images_resized = tf.image.resize_images(images_reshaped, [60,80])
        print("rgb_reshaped size:")
        print(rgb_reshaped.get_shape())
        
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_reshaped)
        #assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        print('split rgb')
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], name="bgr")
        #assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        
        print('concat rgb to bgr')
        
        print('bgr shape:')
        print(bgr.get_shape)
        
        ### VGG Net definition ###
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", trainable=False)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", trainable=False)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", trainable=False)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", trainable=False)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", trainable=False)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable=False)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", trainable=False)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", trainable=False)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", trainable=False)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", trainable=False)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", trainable=False)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", trainable=False)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", trainable=False)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", trainable=False)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", trainable=False)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", trainable=False)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')
        
        # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6", trainable=False)
        self.relu6 = tf.nn.relu(self.fc6)
        # if train_mode is not None:
            # self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        # elif self.trainable:
            # self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        
        if train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", trainable=False)
        self.relu7 = tf.nn.relu(self.fc7)
        # if train_mode is not None:
            # self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        # elif self.trainable:
            # self.relu7 = tf.nn.dropout(self.relu7, self.dropout)
            
        if train_mode:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, num_classes, "fc8_modified", trainable=True)
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

    def conv_layer(self, bottom, in_channels, out_channels, name, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

	######################################################
	# Sets the weights and biases for a convolutional layer
	#  Args:
	#    filter_size: size of conv filter (int, actual filter size is the square of this)
	#    in_size: number of input channels
	#    out_size: number of output channels
	#    name: name of layer
	#  Returns:
	#    filters: filter values for this layer
	#    biases: bias values for this layer
	######################################################
    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return filters, biases

	######################################################
	# Sets the weights and biases for a fully connected layer
	#  Args:
	#    in_size: number of input nodes
	#    out_size: number of output nodes
	#    name: name of layer
	#  Returns:
	#    weights: weight values for this layer
	#    biases: bias values for this layer
	######################################################
    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return weights, biases

		
	######################################################
	# Sets the weights or biases for one layer
	#  Args:
	#    initial_value: truncated normals of filter(for weights) or output(for biases) size
	#    name: name of the layer
	#    idx: 0 for weights, 1 for biases
	#    var_name: name of the entity: 'name' + '_filters' or 'name' + '_biases'
	#  Returns:
	#    var: the variable values read from the data structure (either for the weights or for the biases)
	######################################################
    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
    
    ######################################################
	# Save the updated(?) parameters
	#  Args:
	#    sess: session that runs the functions
	#    npy_path: the file to save the parameters to	 
	#  Returns:
	#    npy_path: returns the path to the file the parameters have been saved to
	######################################################
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path
        

	######################################################
	# Counts the parameters of the model
	#  Args:
	#    None
	#  Returns:
	#    count: number of parameters
	######################################################
    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count