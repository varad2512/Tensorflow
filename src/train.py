import ImgSeg
import tensorflow as tf

from   tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("MNIST_data/" , one_hot = True )


train_obj = ImgSeg.TwoLayerCNN(0,0)
train_obj.graph_build()

print "Training..."

l_cross_entropy    = tf.reduce_mean(-tf.reduce_sum(train_obj.y_true
                     * tf.log(train_obj.y_pred), reduction_indices=[1]))
train_step         = tf.train.AdamOptimizer(1e-4).minimize(l_cross_entropy)


'''correct_prediction = tf.equal(tf.argmax(train_obj.y_pred,1),
                     tf.argmax(train_obj.y_true,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''



summary_tensor     = [["cost_tensor", l_cross_entropy]]
                     #["accuracy_train", accuracy]]
summary_op         = train_obj.summary_write(summary_tensor)
writer             = tf.train.SummaryWriter("/Tensorboard",
                     train_obj.sess.graph)

train_obj.sess.run(tf.initialize_all_variables())
'''
variable_list_save = [train_obj.w_conv1,train_obj.b_conv1,train_obj.w_conv2
                     ,train_obj.b_conv2,train_obj.w_fc1,train_obj.b_fc1,
                      train_obj.w_fc2,train_obj.b_fc2]
saver              =  tf.train.Saver(variable_list_save)
'''
for i in range(2000):
    batch = mnist.train.next_batch(50)
    temp,_,training_accuracy,summary_train = train_obj.sess.run
                                            ([train_obj.w_fc2 ,train_step,
                                            accuracy,summary_op],feed_dict
                                            = {train_obj.x:batch[0],
                                            train_obj.y_true:batch[1]})
    writer.add_summary( summary_train , 200+i )



    '''if(i==1500 ):
        saver.save(train_obj.sess,'Saved_Final.ckpt',i)

print training_accuracy
    '''
