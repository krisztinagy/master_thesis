#Source for reading data: https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
#Source for CNN: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py


# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train and Eval the Treedom network.
This version uses data converted to a TFRecords file containing tf.train.Example 
protocol buffers.
See tensorflow/g3doc/how_tos/reading_data.md#reading-from-files
for context.
YOU MUST run build_image_data.py once before running this.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import datetime
import math

import numpy
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

"""import sys

def trace(frm, e, arg):
    print('%s, %s:%d' % (e, frm.f_code.co_filename, frm.f_lineno))
    return trace

def x(y):
    if y > 0:
        x(y-1)

sys.settrace(trace)
x(15)"""


tf.logging.set_verbosity(tf.logging.DEBUG)


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_epochs_eval', 1, 'Number of epochs to run evaluation.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 2, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'images/output',
                    'Directory with the training data.')
flags.DEFINE_integer('image_pixels', 4800, 'Number of pixels in image')
#flags.DEFINE_integer('image_height', 60, 'Image height')
#flags.DEFINE_integer('image_width', 80, 'Image width')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
#flags.DEFINE_integer('num_channels', 1, 'Number of image channels')
#flags.DEFINE_integer('seed', 66478, 'Python integer used to create a random seed for the distribution')

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'small-train-00000-of-00001'
VALIDATION_FILE = 'small-validation-00000-of-00001'

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
  	serialized_example,
  	features={
      'image/height':tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'image/colorspace': tf.FixedLenFeature([], tf.string),
      'image/channels': tf.FixedLenFeature([], tf.int64),
      'image/class/label': tf.FixedLenFeature([], tf.int64),
      'image/class/text': tf.FixedLenFeature([], tf.string),
      'image/format': tf.FixedLenFeature([], tf.string),
      'image/filename': tf.FixedLenFeature([],tf.string),
      'image/encoded': tf.FixedLenFeature([], tf.string),
  })

  # Convert from a scalar string tensor to a uint8 tensor with shape [FLAGS.image_pixels].
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  print("BEFORE SETTING")
  print(image.get_shape())
  image.set_shape([FLAGS.image_pixels])
  
  #Reshape into a Tensor of shape [height, width, channels]
  #image_reshaped = tf.reshape(image, (FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels))
  #print("reshaped size:")
  #print(image_reshaped.get_shape())

  # OPTIONAL: Apply distortion here

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  
  print("IMAGE SHAPE")
  print(image.get_shape())

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1
  print("LABEL SHAPE")
  print(label.get_shape())

  return image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, FLAGS.num_classes).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  print("enter")
  print(train)
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into FLAGS.batch_size batches.
    # (Internally uses a RandomShuffleQueue.)We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    print("images shape:")
    print(images.get_shape())
    return images, sparse_labels

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


def inference_graph(images, train):

	### Dimension info: images shape: unknown at this point since images is a placeholder
	### Dimension info: images shape: expected to be [100, 4800] for a batch size of 100 and image size 60*80*1
	print("images dim:")
	print(images.get_shape())
	
	### Dimension info: images_reshaped shape: expected to be [100, 60, 80, 1]
	images_reshaped = tf.reshape(images, [-1,60,80,1])
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
	
	W_fc1 = weight_variable([15 * 20 * 64, FLAGS.hidden1])
	print(W_fc1.get_shape())
	b_fc1 = bias_variable([FLAGS.hidden1])
	
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
		
	W_fc2 = weight_variable([FLAGS.hidden1, FLAGS.num_classes])
	b_fc2 = bias_variable([FLAGS.num_classes])
	
	### Dimension info: logits shape: expected to be [100, 2]
	logits = tf.matmul(h_fc1, W_fc2) + b_fc2
	print("logits shape:")
	print(logits.get_shape())
	
	return logits
	
	

"""Build the training graph.
   Args:
       logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
       labels: Labels tensor, int32 - [BATCH_SIZE], with values in the
         range [0, NUM_CLASSES).
       learning_rate: The learning rate to use for gradient descent.
   Returns:
       train_op: The Op for training.
       loss: The Op for calculating loss.
"""
"""def training_graph(logits, labels, learning_rate):

    print("a")
    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    print("b")
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
    print("c")
    # Create a variable to track the global step (iteration).
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    print("d")
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    print("e")
    return train_op, loss"""
    
def training_graph(logits, labels, learning_rate):

    # Create an operation that calculates loss.
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    
    # Create a variable to track the global step (iteration).
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op, loss
    
def run_training():
	# Tell TensorFlow that the model will be built into the default Graph.
	treedom_graph=tf.Graph()
	with treedom_graph.as_default():
	
		#Generate placeholders for images and labels
	  	#Ensures that the same graph can be used for training, inference and evaluation later.
	    images_placeholder=tf.placeholder(tf.float32)
	    labels_placeholder=tf.placeholder(tf.int32)
	    print("PH Images shape:")
	    print(images_placeholder.shape)
	    print("PH Labels shape:")
	    print(labels_placeholder.shape)
	    
	    #Remember these operands
	    tf.add_to_collection("images", images_placeholder)
	    tf.add_to_collection("labels", labels_placeholder)
	    
	    # Build a Graph that computes predictions from the inference model.
	    logits = inference_graph(images_placeholder, True)
	    
	    #remember this operation
	    tf.add_to_collection("logits", logits)
	    
	    #create images and labels
	    images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
	    #images_eval, labels_eval = inputs(train=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs_eval)
	    print("printing images and labels")
	    print(images)
	    print(labels)
	    
	    #create train and loss op
	    train_op, loss =training_graph(logits, labels_placeholder, FLAGS.learning_rate)
	    	    
	    #ititalize global and local variables
	    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	    
	    #Create a saver for writing training checkpoints
	    saver=tf.train.Saver()
	    print("end")
	    
	with tf.Session(graph=treedom_graph) as sess:
	
		print("before init")
		#initialize all the variables by running the op
		sess.run(init)
		
		print("after init")
		
		#variable for tracking losses - to be displayed in losses.png
		losses = []
		
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		print("1")
		
		iteration = 0
		last_loss = 1
		try:
			while not coord.should_stop():
				start_time = time.time()
				
				image, label = sess.run([images, labels])
					
				_, loss_value = sess.run([train_op, loss], feed_dict={images_placeholder: image, labels_placeholder:label})
				last_loss = loss_value
				
				
				print("Step: %d, Loss:%.2f" % (iteration, loss_value))
				#print("4")
				
				losses.append(loss_value)
				
				duration = time.time() - start_time
				
				# Print an overview fairly often.
				if iteration % 100 == 0:
					print('Step %d: loss = %.2f (%.3f sec)' % (iteration, loss_value, duration))
				
				"""
				if(step + 1) % 1000 == 0:
					print('Saving checkpoint...')
					checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
					saver.save(sess, checkpoint_file, global_step=step)"""
				
				iteration += 1
				
		except tf.errors.OutOfRangeError:
			print('Done training for %d epochs, %d iterations.' % (FLAGS.num_epochs, iteration))
			f.write('Done training for %d epochs, %d iterations.\n' % (FLAGS.num_epochs, iteration))
			f.write('Final loss value: %.3f\n' % (last_loss))
		finally:
			coord.request_stop()
			
		# Wait for threads to finish.
		coord.join(threads)

		saver.save(sess, 'images/output/my-small-model-1000')
		saver.export_meta_graph('images/output/my-small-model-1000.meta')
		
		tf.train.write_graph(tf.get_default_graph().as_graph_def(), "/tmp", "exported.pbtxt", as_text=True)
		
		fig=plt.figure()
		a1=fig.add_subplot(111)
		a1.plot(losses, label="losses")
		fig.savefig("losses.png")
		
		sess.close

f = open('results_log', 'a')
f.write('----TRAINING----\n')
start_time=datetime.datetime.now()
f.write('Started training: ')
f.write(str(start_time))
f.write('\n')
run_training()
end_time=datetime.datetime.now()
f.write('Finished training: ')
f.write(str(end_time))
f.write('\nTraining took: ')
f.write(str(end_time - start_time))
f.write('\n')
f.close()