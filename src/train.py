#import ImgSeg
import tensorflow as tf
import FCN_32
from Import import *
'''
from   tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("MNIST_data/" , one_hot = True )
'''

train_obj = FCN_32.FCN()
train_obj.graph_build()

print "Training..."

reshaped_logits = tf.reshape(train_obj.transpose_conv, [-1, 21])  # shape [batch_size*256*256, 21]
reshaped_labels = tf.reshape(train_obj.y_true, [-1])              # shape [batch_size*256*256]
loss            = tf.nn.sparse_softmax_cross_entropy_with_logits(reshaped_logits,
                                              reshaped_labels,name="cost_tensor")
train_step      = tf.train.AdamOptimizer(1e-4).minimize(loss)

loss            = tf.reduce_mean(loss)

'''
l_cross_entropy    = tf.reduce_mean(-tf.reduce_sum(train_obj.y_true
                     * tf.log(train_obj.y_pred), reduction_indices=[1]))
train_step         = tf.train.AdamOptimizer(1e-4).minimize(l_cross_entropy)
'''

'''correct_prediction = tf.equal(tf.argmax(train_obj.y_pred,1),
                     tf.argmax(train_obj.y_true,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''



#summary_tensor     = [["cost_tensor", loss]]
tf.scalar_summary("cost_train",loss)
                     #["accuracy_train", accuracy]]
#summary_op         = train_obj.summary_write(summary_tensor)
summary_op         = tf.merge_all_summaries()
writer             = tf.train.SummaryWriter("/home/varad/work/Tensorflow/Tensorboard",
                     train_obj.sess.graph)

#tf.scalar("loss",loss)

'''
variable_list_save = [train_obj.w_conv1,train_obj.b_conv1,train_obj.w_conv2
                     ,train_obj.b_conv2,train_obj.w_fc1,train_obj.b_fc1,
                      train_obj.w_fc2,train_obj.b_fc2]
saver              =  tf.train.Saver(variable_list_save)
'''
train_obj.sess.run(tf.initialize_all_variables())

for j in range(1):
    for i in range(5):
        #batch = mnist.train.next_batch(50)
        image,label = next()
        summary_accuracy,k,l,temp,_=train_obj.sess.run([summary_op,train_obj.transpose_conv,
                                    loss,train_obj.h_FC3,train_step],
                                    feed_dict = {train_obj.x:image, train_obj.y_true:label})
        print i

        #print k.shape
        #print l.shape
        writer.add_summary(summary_accuracy,i)


'''
        temp = []
        temp = k[0,:,:,:]

        print temp.shape
        from PIL import Image
        img = Image.fromarray(temp, 'RGB')
        img.save('my.png')
        img.show()

        print training_accuracy
'''
