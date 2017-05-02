"""Evalution of the neural network.
This version uses data converted to a TFRecords file containing tf.train.Example 
protocol buffers.
See tensorflow/g3doc/how_tos/reading_data.md#reading-from-files for context.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'loss_functions')
import time
import datetime
import math
import shutil

import numpy as np
import tensorflow as tf
import config as cfg
exec "import %s as model" % ( cfg.model['model_import'] )
exec "import %s as loss_function" % ( cfg.model['loss_function_import'] )

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', cfg.hyperparameters['learning_rate'], 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', cfg.hyperparameters['num_epochs'], 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_epochs_eval', cfg.hyperparameters['num_epochs_eval'], 'Number of epochs to run evaluation.')
flags.DEFINE_integer('batch_size', cfg.hyperparameters['batch_size'], 'Batch size.')
flags.DEFINE_integer('image_pixels', cfg.image['height'] * cfg.image['width'] * cfg.image['channels'], 'Number of pixels in image')

#flags.DEFINE_integer('hidden1', cfg.model['hidden1'], 'Number of units in hidden layer 1.')
#flags.DEFINE_integer('hidden2', cfg.model['hidden2'], 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_classes', cfg.dataset['num_categories'], 'Number of classes')

flags.DEFINE_string('train_dir','train-' + str(cfg.dataset['training_set_size']) + '-test-' + str(cfg.dataset['testing_set_size']) + '/' + cfg.directory['output'] + '/', 'Directory with the training data.')
flags.DEFINE_string('training_data_file', cfg.tfRecords['prefix'] + '-' + cfg.tfRecords['training_file'], 'Training data file')
flags.DEFINE_string('evaluation_data_file', cfg.tfRecords['prefix'] + '-' + cfg.tfRecords['testing_file'], 'Training data file')
flags.DEFINE_string('model_dir', 'flat-e%d/' % cfg.hyperparameters['num_epochs'], 'Directory for storing model and results')


######################################################
# Reads tfRecords files
#  Args:
#    filename_queue: the queue containing the files to be processed
#  Returns:
#    image: flattened image data tf.float32 [-0.5 0.5]
#	 label: the label of the processed image, tf.int32 (0, 1, etc.)
######################################################
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

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar and handle offset
  label = tf.cast(features['image/class/label'], tf.int32) - 1

  return image, label

######################################################
# Reads input data num_epochs times.
#  Args:
#    train: Selects between the training (True) and validation (False) data.
#	 batch_size: Number of examples per returned batch.
#	 num_epochs: Number of times to read the input data, or 0/None to train forever.
#  Returns:
#    A tuple (images, labels), where:
#    * images is a float tensor with shape [batch_size, FLAGS.image_pixels] in [-0.5, 0.5].
#    * labels is an int32 tensor with shape [batch_size] with the true label, in [0, FLAGS.num_classes).
#    Note that an tf.train.QueueRunner is added to the graph, which must be run using e.g. tf.train.start_queue_runners().
######################################################
def inputs(train, batch_size, num_epochs):

  if not num_epochs: num_epochs = None
  filename = os.path.join(FLAGS.train_dir,
                          FLAGS.training_data_file if train else FLAGS.evaluation_data_file)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename queue
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into FLAGS.batch_size batches. Internally uses a RandomShuffleQueue.)
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

###INFERENCE GRAPH - implemented in separate python module in folder 'models'

###TRAINING GRAPH - implemented in separate python module in folder 'loss_functions'

######################################################
######################################################
################      EVALUATION       ###############
######################################################
######################################################
def run_evaluation():

	treedom_graph=tf.Graph()
	with treedom_graph.as_default():
	
		# Define placeholders
		images_placeholder=tf.placeholder(tf.float32)
		labels_placeholder=tf.placeholder(tf.int32)
		# Remember these operands
		tf.add_to_collection("images", images_placeholder)
		tf.add_to_collection("labels", labels_placeholder)
		
		# Build a Graph that computes predictions from the inference model.
		logits = model.inference_graph(images_placeholder, FLAGS.image_pixels, FLAGS.num_classes)
		tf.add_to_collection("logits", logits)
		
		#create images and labels
		images_eval, labels_eval = inputs(train=False, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs_eval)
		
		#create eval op
		all = tf.nn.top_k(logits)
		eval_op_values, eval_op_indices = tf.nn.top_k(logits)
		
		#ititalize global and local variables
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		
	with tf.Session(graph=treedom_graph) as sess:		
		
		sess.run(init)
		
		checkpoint = FLAGS.train_dir + FLAGS.model_dir + 'check'
		
		saver=tf.train.Saver()
		saver.restore(sess, checkpoint)
		
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
		
		step = 0
		total_error = 0
		
		# Start input enqueue threads.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
		try:
			while not coord.should_stop():
				
				start_time = time.time()
				
				print("Threading")
				
				image, label = sess.run([images_eval, labels_eval])
				
				all_data, logit, pred_values, prediction = sess.run([all, logits, eval_op_values, eval_op_indices], feed_dict={images_placeholder: image, labels_placeholder: label})
				
				print('all:')
				print(all_data)
				print('logit:')
				print(logit)
				print('prediction')
				print(pred_values)
				print('index')
				print(prediction)
	
				prediction = prediction.flatten()
				
				labels = labels + label.tolist()
				predictions = predictions + prediction.tolist()
				
				duration = time.time() - start_time
				
				abs = np.absolute(label - prediction)
				
				error=np.sum(abs)
				
				print("Error: %.3f Step: %d" % (error, step))
					
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

		print("Average error during evaluation of %d steps: %.3f" % (step, total_error))
		f.write("Average error during evaluation of %d steps: %.3f\n" % (step, total_error))

		# Wait for threads to finish.
		coord.join(threads)
		
		print(predictions)
		print(labels)
		
		difference = (np.asarray(labels) - np.asarray(predictions)) / 2
		
		print(difference)
		
		difference = difference.tolist()
		
		print(difference)
		
		fig=plt.figure()
		a=fig.add_subplot(111)
		a.plot(labels, 'bo', label="labels")
		a.plot(predictions, 'go', label="predictions")
		a.plot(difference, 'ro', label="difference")
		#plt.legend()
		fig.savefig( FLAGS.train_dir + FLAGS.model_dir + 'flat_classification_evaluation.png' )
		
		difference = np.absolute(difference *2)
		
		sess.close()
		
directory = FLAGS.train_dir + FLAGS.model_dir
print(directory)
if not os.path.exists(directory):
	os.makedirs(directory)
	
shutil.copy2('/home/nagy729krisztina/Preprocessing/config.py', directory)

f = open( directory + 'log', 'a+')
f.write('\n\n----FLAT CLASSIFICATION EVALUATION----\n\n')
f.write('Batch size: %d\n' % (FLAGS.batch_size))
f.write('Number of epochs: %d\n' % (FLAGS.num_epochs))
f.write('Image size: %d\n\n' % (FLAGS.image_pixels))
f.write('--Model information--\n')
f.write('Model: %s\n' % (cfg.model['model_import']))
f.write('Loss function: %s\n\n' % (cfg.model['loss_function_import']))
f.write('--Results--\n')

run_evaluation()

start_time=datetime.datetime.now()
f.write('\nStarted training: ')
f.write(str(start_time))
f.write('\n')
end_time=datetime.datetime.now()
f.write('Finished training: ')
f.write(str(end_time))
f.write('\nTraining took: ')
f.write(str(end_time - start_time))
f.write('\n\nTraining ended successfully\n')
f.close()	
