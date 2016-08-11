#vgg 16 implementation
import tensorflow as tf
from init import *
from activation import *
from convolute import *
from max_pool import *

class FCN():

    def __init__(cnn_base.CNNBase):
        cnn_base.CNNBase.__init__(self)


    def build(self):

        self.x            = tf.placeholder(tf.float32 ,name = "Input")
        self.y_true       = tf.placeholder(tf.float32 ,name = "Output")
        x_image           = tf.reshape(self.x, [-1,tf.shape(self.x)[1],tf.shape(self.x)[2],tf.shape(self.x)[3]])

        self.w_conv1      = self.weight_init([3,3,1,64],"weight1")
        self.b_conv1      = self.bias_init([64],"bias1")
        h_conv1           = self.convolve_activate(x_image, self.w_conv1) + self.b_conv1))

        self.w_conv2      = self.weight_init([3,3,64,64],"weight2")
        self.b_conv2      = self.bias_init([64],"bias2")
        h_conv2           = self.convolve_activate(h_conv1, self.w_conv2) + self.b_conv2))
        h_pool2           = max_pool_2x2(h_conv2)

        self.w_conv3      = self.weight_init([3,3,64,128],"weight3")
        self.b_conv3      = self.bias_init([128],"bias3")
        h_conv3           = self.convolve_activate(h_pool2, self.w_conv3) + self.b_conv3))

        self.w_conv4      = self.weight_init([3,3,128,128],"weight4")
        self.b_conv4      = self.bias_init([128],"bias4")
        h_conv4           = self.convolve_activate(h_conv3, self.w_conv4) + self.b_conv4))
        h_pool4           = max_pool_2x2(h_conv4)


        self.w_conv5      = self.weight_init([3,3,128,256],"weight5")
        self.b_conv5      = self.bias_init([256],"bias5")
        h_conv5           = self.convolve_activate(h_pool4, self.w_conv5) + self.b_conv5))


        self.w_conv6      = self.weight_init([3,3,256,256],"weight6")
        self.b_conv6      = self.bias_init([256],"bias6")
        h_conv6           = self.convolve_activate(h_conv5, self.w_conv6) + self.b_conv6))


        self.w_conv7      = self.weight_init([1,1,256,256],"weight7")
        self.b_conv7      = self.bias_init([256],"bias7")
        h_conv7           = self.convolve_activate(h_conv6, self.w_conv7) + self.b_conv7))
        h_pool7           = max_pool_2x2(h_conv7)

        self.w_conv8      = self.weight_init([3,3,256,512],"weight8")
        self.b_conv8      = self.bias_init([512],"bias8")
        h_conv8           = self.convolve_activate(h_pool7, self.w_conv8) + self.b_conv8))


        self.w_conv9      = self.weight_init([3,3,512,512],"weight9")
        self.b_conv9      = self.bias_init([512],"bias9")
        h_conv9           = self.convolve_activate(h_conv8, self.w_conv9) + self.b_conv9))


        self.w_conv10     = self.weight_init([1,1,512,512],"weight10")
        self.b_conv10     = self.bias_init([512],"bias10")
        h_conv10          = self.convolve_activate(h_conv9, self.w_conv10) + self.b_conv10))
        h_pool10          = max_pool_2x2(h_conv10)

        self.w_conv11     = self.weight_init([3,3,512,512],"weight11")
        self.b_conv11     = self.bias_init([512],"bias11")
        h_conv11          = self.convolve_activate(h_pool10, self.w_conv11) + self.b_conv11))

        self.w_conv12     = self.weight_init([3,3,512,512],"weight12")
        self.b_conv12     = self.bias_init([512],"bias12")
        h_conv12          = self.convolve_activate(h_conv11, self.w_conv12) + self.b_conv12))


        self.w_conv13     = self.weight_init([1,1,512,512],"weight13")
        self.b_conv13     = self.bias_init([512],"bias13")
        h_conv13          = self.convolve_activate(h_conv12, self.w_conv13) + self.b_conv13))
        h_pool13          = max_pool_2x2(h_conv13)



        self.weight_FC1   =  self.weight_init([7,7,512, 4096] , "weight_FC1")
        self.bias_FC1     =  self.bias_init([4096] , "bias_FC1")
        h_FC1             =  tf.nn.relu(conv2d(h_pool13,self.weight_FC1) + self.bias_FC1)

        self.weight_FC2   =  self.weight_init([1,1,4096,4096] , "weight_FC2")
        self.bias_FC2     =  self.bias_init([4096] , "bias_FC2")
        h_FC2             =  tf.nn.relu(conv2d(h_FC1,self.weight_FC2) + self.bias_FC2)


        self.weight_FC3   =  self.weight_init([1,1,4096,1024] , "weight_FC3")
        self.bias_FC3     =  self.bias_init([1024] , "bias_FC3")
        h_FC3             =  conv2d(h_FC2,self.weight_FC3) + self.bias_FC3     #skip relu for last 1*1 conv layer

        self.weight_deconv = self.weight_init([64,64,21,1024] , "weight_deconv")

        self.transpose_conv =  tf.nn.conv2d_transpose(h_FC3, self.weight_deconv , [tf.shape(self.x)[0],tf.shape(self.x)[1],tf.shape(self.x)[2],21], [1,32,32,1], padding='VALID', name="DECONVOLUTION")
