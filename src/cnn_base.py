  # Base class for Convolutional Neural Network. Inherit this class for models that
# are to be created.
from math import *
import numpy as np
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

  	def _upscore_layer(self, bottom, shape,
                     num_classes, name,
                     ksize=64, stride=32):
	        strides = [1, stride, stride, 1]
	        with tf.variable_scope(name):
	            in_features = bottom.get_shape()[3].value

	            if shape is None:
	                # Compute shape out of Bottom
	                in_shape = tf.shape(bottom)

	                h = ((in_shape[1] - 1) * stride) + 1
	                w = ((in_shape[2] - 1) * stride) + 1
	                new_shape = [in_shape[0], h, w, num_classes]
	            else:
	                new_shape = [shape[0], shape[1], shape[2], num_classes]
	            output_shape = tf.pack(new_shape)


	            f_shape = [ksize, ksize, num_classes, in_features]

	            # create
	            num_input = ksize * ksize * in_features / stride
	            stddev = (2 / num_input)**0.5

	            weights = self.get_deconv_filter(f_shape)
	            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
	                                            strides=strides, padding='SAME')

	        return deconv

	def get_deconv_filter(self, f_shape):
		        width = f_shape[0]
		        heigh = f_shape[0]
		        f = ceil(width/2.0)
		        c = (2 * f - 1 - f % 2) / (2.0 * f)
		        bilinear = np.zeros([f_shape[0], f_shape[1]])
		        for x in range(width):
		            for y in range(heigh):
		                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
		                bilinear[x, y] = value
		        weights = np.zeros(f_shape)
		        for i in range(f_shape[2]):
		            weights[:, :, i, i] = bilinear

		        init = tf.constant_initializer(value=weights,
		                                       dtype=tf.float32)
		        return tf.get_variable(name="up_filter", initializer=init,
		                               shape=weights.shape)



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
