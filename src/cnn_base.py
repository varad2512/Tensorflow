  # Base class for Convolutional Neural Network. Inherit this class for models that
# are to be created.

import tensorflow as tf
import abc
class CNNBase():

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		tf.reset_default_graph()
		self.sess = tf.InteractiveSession()

	# Initialize weights with random values picked from a distribution with
    #"0" mean and "0.1"
	# standard deviation
	def weight_init(self, shape, name):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial, name = name)

	# Initialize bias to a constant value
	def bias_init(self, shape, name):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial, name = name)

	# Convolve input and filter. Add Bias
	def convolve_activate(self, input, filter, bias, strides=[1,1,1,1], padding='SAME'):
		conv_output	      = tf.nn.conv2d(input, filter, strides, padding) + bias
		activation_output = tf.nn.relu(conv_output)
		return activation_output

	# Create Summary Protocol Buffer.
	# Args:
	# tag_value_pair: A tensor of rank 2 and shape [n,2];n is
	# the number of different tensors to be logged
	# Return:
	# Summary protocol buffer
	def summary_write(self, tag_value_pair):
		for tag_value in tag_value_pair:
			tf.scalar_summary(tag_value[0],tag_value[1])
		summary_op = tf.merge_all_summaries()
		return summary_op

	def loss_calc(self):
		pass

	@abc.abstractmethod
	def graph_build(self):
		# Implementation should build the tensorflow graph for the corresponding
        #network model
		return
