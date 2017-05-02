import tensorflow as tf
import config as cfg
import math

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference_graph(images, train, image_pixels, num_classes):

	hidden1_units = 64
	
	### Dimension info: images shape: unknown at this point since images is a placeholder
	### Dimension info: images shape: expected to be [100, 4800] for a batch size of 100 and image size 60*80*1
	print("images dim:")
	print(images.get_shape())
	
	### Dimension info: images_reshaped shape: expected to be [100, 60, 80, 1]
	images_reshaped = tf.reshape(images, [-1,cfg.image['height'],cfg.image['width'],1])
	#images_resized = tf.image.resize_images(images_reshaped, [60,80])
	print("images_reshaped size:")
	print(images_reshaped.get_shape())
	
	#To compute 32 features for each 5*5 patch - dimensions: [patch size, patch size, input channels, output channels]
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	### Dimension info: h_conv1 shape: expected to be [100, 60, 80, 32]
	h_conv1 = tf.nn.relu(conv2d(images_reshaped, W_conv1) + b_conv1)
	print("conv1 shape:")
	print(h_conv1.get_shape())
	
	### Dimension info: h_pool1 shape: expected to be [100, 30, 40, 32]
	h_pool1 = max_pool_2x2(h_conv1)
	print("pool1 shape:")
	print(h_pool1.get_shape())
	
	#To compute 64 features for each 5*5 patch - dimensions: [patch size, patch size, input channels, output channels]
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	
	### Dimension info: h_conv2 shape: expected to be [100, 30, 40, 64]
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	print("conv2 shape:")
	print(h_conv2.get_shape())
	
	### Dimension info: h_pool2 shape: expected to be [100, 15, 20, 64]
	h_pool2 = max_pool_2x2(h_conv2)
	print("pool2 shape:")
	print(h_pool2.get_shape())
	
	W_fc1 = weight_variable([15 * 20 * 64, hidden1_units])
	print(W_fc1.get_shape())
	b_fc1 = bias_variable([hidden1_units])
	
	### Dimension info: h_pool2_flat shape: expected to be [100, 19200]
	h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 20 * 64])
	print("pool2_flat shape:")
	print(h_pool2_flat.get_shape())
	
	### Dimension info: h_fc1 shape: expected to be [100, 128]
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	print("fully connected shape:")
	print(h_fc1.get_shape())
	
	#if train:
	#	h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
		
	W_fc2 = weight_variable([hidden1_units, num_classes])
	b_fc2 = bias_variable([num_classes])
	
	### Dimension info: logits shape: expected to be [100, 2]
	logits = tf.matmul(h_fc1, W_fc2) + b_fc2
	print("logits shape:")
	print(logits.get_shape())
	
	return logits