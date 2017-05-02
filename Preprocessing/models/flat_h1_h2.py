import tensorflow as tf
import math

######################################################
# Build the model up to where it may be used for inference.
#  Args:
#    images: Images placeholder.
#	 hidden1_units: Size of the first hidden layer.
#	 hidden2_units: Size of the second hidden layer
#  Returns:
#    logits: Output tensor with the computed logits.
######################################################


def inference_graph(images, image_pixels, num_classes):

	hidden1_units = 62
	hidden2_units = 2

	# Hidden 1
	with tf.name_scope('hidden1'):
	    weights = tf.Variable(
	        tf.truncated_normal([image_pixels, hidden1_units],
	                            stddev=1.0 / math.sqrt(float(image_pixels))),
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
		    tf.truncated_normal([hidden2_units, num_classes],
		                        stddev=1.0 / math.sqrt(float(hidden2_units))),
	        name='weights')
	    biases = tf.Variable(tf.zeros([num_classes]),
		                     name='biases')
	    logits = tf.matmul(hidden2, weights) + biases
	    
	return logits