# Two Layer CNN with fully connected networks for MNIST_data

import cnn_base
import tensorflow as tf

class TwoLayerCNN(cnn_base.CNNBase):

    def __init__(self, debug, saver):
        DEBUG = debug
        SAVER = saver
        cnn_base.CNNBase.__init__(self)

    def max_pool_2x2(self, input):
        return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding = 'SAME')






    def graph_build(self):
        self.x       = tf.placeholder( tf.float32 , shape = [None,784] ,
                       name = "Input" )
        self.y_true  = tf.placeholder( tf.float32 , shape = [None,10]  ,
                       name = "Ouput_true" )
        x_image      = tf.reshape(self.x, [-1,28,28,1])

        self.w_conv1 = self.weight_init([5,5,1,32], "weight1")
        self.b_conv1 = self.bias_init([32], "bias1")
        h_conv1      = self.convolve_activate(x_image, self.w_conv1,
                       self.b_conv1)
        h_pool1      = self.max_pool_2x2(h_conv1)

        self.w_conv2 = self.weight_init([5,5,32,64],"weight2")
        self.b_conv2 = self.bias_init([64],"bias2")
        h_conv2      = self.convolve_activate(h_pool1, self.w_conv2,
                       self.b_conv2)
        h_pool2      = self.max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        self.w_fc1   = self.weight_init([7*7*64, 1024],"weight3")
        self.b_fc1   = self.bias_init([1024],"bias3")
        h_fc1        = tf.nn.relu((tf.matmul(h_pool2_flat, self.w_fc1) +
                       self.b_fc1))

        self.w_fc2   = self.weight_init([1024, 10],"weight4")
        self.b_fc2   = self.bias_init([10],"bias4")
        self.y_pred  = tf.nn.softmax(tf.matmul(h_fc1, self.w_fc2) +
                       self.b_fc2)
