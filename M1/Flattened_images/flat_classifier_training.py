#Source: https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

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


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_epochs_eval', 1, 'Number of epochs to run evaluation.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'images/output',
                    'Directory with the training data.')
flags.DEFINE_integer('image_pixels', 47000, 'Number of pixels in image')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train-00000-of-00001'
VALIDATION_FILE = 'validation-00000-of-00001'


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

  # Convert from a scalar string tensor to a uint8 tensor with shape
  # [FLAGS.image_pixels].
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image.set_shape([FLAGS.image_pixels])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1

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
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
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
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

"""Build the model up to where it may be used for inference.
Args:
        images: Images placeholder.
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
    Returns:
        logits: Output tensor with the computed logits.
"""
def inference_graph(images, hidden1_units, hidden2_units):

	# Hidden 1
	with tf.name_scope('hidden1'):
	    weights = tf.Variable(
	        tf.truncated_normal([FLAGS.image_pixels, hidden1_units],
	                            stddev=1.0 / math.sqrt(float(FLAGS.image_pixels))),
	        name='weights')
	    biases = tf.Variable(tf.zeros([hidden1_units]),
	                         name='biases')
	    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
	    
	# Hidden 2
	with tf.name_scope('hidden2'):
	    weights = tf.Variable(
		    tf.truncated_normal([hidden1_units, hidden2_units],
		                        stddev=1.0 / math.sqrt(float(hidden1_units))),
	        name='weights')
	    biases = tf.Variable(tf.zeros([hidden2_units]),
		                     name='biases')
	    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
	    
	# Linear
	with tf.name_scope('softmax_linear'):
	    weights = tf.Variable(
		    tf.truncated_normal([hidden2_units, FLAGS.num_classes],
		                        stddev=1.0 / math.sqrt(float(hidden2_units))),
	        name='weights')
	    biases = tf.Variable(tf.zeros([FLAGS.num_classes]),
		                     name='biases')
	    logits = tf.matmul(hidden2, weights) + biases
	    
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
	    logits = inference_graph(images_placeholder,
	                             FLAGS.hidden1,
	                             FLAGS.hidden2)
	    
	    #remember this operand
	    tf.add_to_collection("logits", logits)
	    
	    #create images and labels
	    images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
	    #images_eval, labels_eval = inputs(train=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs_eval)
	    print("printing images and labels")
	    print(images)
	    print(labels)
	    
	    #create train and loss op
	    train_op, loss =training_graph(logits, labels_placeholder, FLAGS.learning_rate)
	    
	    #create eval op
	    #eval_op_values, eval_op_indices = tf.nn.top_k(logits)
	    
	    #ititalize global and local variables
	    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	    
	    #Create a saver for writing training checkpoints - save the state of the network
	    #So that later we can evaluate the network, or continue training if it was interrupted
	    #instead of initializing again and starting from scratch
	    saver=tf.train.Saver()
	    
	with tf.Session(graph=treedom_graph) as sess:
	
		#initialize all the variables by running the op
		sess.run(init)
		
		# Input images and labels.
		#images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
		
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

		saver.save(sess, 'images/output/my-model-1000')
		saver.export_meta_graph('images/output/my-model-1000.meta')
		
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