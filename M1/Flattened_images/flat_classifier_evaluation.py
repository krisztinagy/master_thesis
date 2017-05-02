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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import datetime
import math

import numpy
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
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
  
  image_reshaped = tf.reshape(image, (features['image/height'], features['image/width'], features['image/channel']))
  image_reshaped.set_shape((features['image/height'], features['image/width'], features['image/channel']))

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image_reshaped = tf.cast(image_reshaped, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1

  return image_reshaped, label


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
    
def run_evaluation():

	treedom_graph=tf.Graph()
	with treedom_graph.as_default():
	
		images_placeholder=tf.placeholder(tf.float32)
		labels_placeholder=tf.placeholder(tf.int32)
		
		#Remember these operands
		tf.add_to_collection("images", images_placeholder)
		tf.add_to_collection("labels", labels_placeholder)
		
		# Build a Graph that computes predictions from the inference model.
		logits = inference_graph(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
		tf.add_to_collection("logits", logits)
		
		#create images and labels
		images_eval, labels_eval = inputs(train=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs_eval)
		
		#create eval op
		eval_op_values, eval_op_indices = tf.nn.top_k(logits)
		
		#ititalize global and local variables
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		
	with tf.Session(graph=treedom_graph) as sess:		
		
		print("enter")
		sess.run(init)
		##saver=tf.train.import_meta_graph(os.path.join(FLAGS.train_dir, "checkpoint-10.meta"))
		saver=tf.train.Saver()
		saver.restore(sess, os.path.join(FLAGS.train_dir, "my-model-1000"))
		#new_saver = tf.train.import_meta_graph('images/output/my-model-1000.meta')
		#new_saver.restore(sess, 'images/output/my-model-1000')
		
		#tf.train.write_graph(tf.get_default_graph().as_graph_def(), "/tmp", "imported.pbtxt", as_text=True)
				
		logits = tf.get_collection("logits")[0]
		print("logits:")
		print(logits)
		
		images_placeholder = tf.get_collection("images")[0]
		print("images")
		print(images_placeholder)
		
		labels_placeholder = tf.get_collection("labels")[0]
		print("labels")
		print(labels_placeholder)
		
		#variable for tracking losses - to be displayed in losses.png
		predictions = []
		labels = []
		
		print("6")
		
		step = 0
		total_error = 0
		
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		print("7")
	
		try:
			while not coord.should_stop():
				start_time = time.time()
				
				print("8")
				
				image, label = sess.run([images_eval, labels_eval])
				
				print("8.5")
				
				labels.append(label)
				
				print(logits)
				
				print("9")
				
				_, prediction = sess.run([eval_op_values, eval_op_indices], feed_dict={images_placeholder: image, labels_placeholder: label})
	
				prediction = prediction.flatten()
				
				predictions.append(prediction)
				
				duration = time.time() - start_time
				
				#print("label:")
				#print(label)
				#print("prediction:")
				#print(prediction)
				
				abs = numpy.absolute(label - prediction)
	
				#print("absolute:")
				#print(abs)
				
				error=numpy.sum(abs)
				
				print(error)
					
				step += 1
				total_error += error
				
		except tf.errors.OutOfRangeError:
			print('Done evaluation for %d epochs, %d steps.' % (FLAGS.num_epochs_eval, step))
			f.write('Done evaluation for %d epochs, %d steps.\n' % (FLAGS.num_epochs_eval, step))
			print("done")
		finally:
			# When done, ask the threads to stop.
			coord.request_stop()
			
		total_error = total_error / ( FLAGS.batch_size * step )

		print("Error rate during evaluation of %d steps: %.3f" % (step, total_error))
		f.write("Error rate during evaluation of %d steps: %.3f\n" % (step, total_error))

		# Wait for threads to finish.
		coord.join(threads)
		sess.close()
		
		fig2=plt.figure()
		a2=fig2.add_subplot(111)
		a2.plot(labels, label="labels")
		a2.plot(predictions, label="predictions")
		plt.legend()
		fig2.savefig("label_prediction.png")
		


f = open('results_log', 'a')
f.write('----EVALUATION----\n')
start_time=datetime.datetime.now()
f.write('Started evaluation: ')
f.write(str(start_time))
f.write('\n')
run_evaluation()
end_time=datetime.datetime.now()
f.write('Finished evaluation: ')
f.write(str(end_time))
f.write('\nEvaluation took: ')
f.write(str(end_time - start_time))
f.write('\n')
f.close()	