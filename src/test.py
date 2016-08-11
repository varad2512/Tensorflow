import ImgSeg

from    tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True )

test_obj = ImgSeg.TwoLayerCNN(0,0)
test_obj.graph_build()

print "Testing..."

correct_prediction = tf.equal(tf.argmax(test_obj.y_pred,1),
                     tf.argmax(test_obj.y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

variable_list_restore=[test_obj.w_conv1,test_obj.b_conv1,test_obj.w_conv2,
                      test_obj.b_conv2,test_obj.w_fc1,test_obj.b_fc1,
                      test_obj.w_fc2,test_obj.b_fc2]
saver = tf.train.Saver(variable_list_restore)

checkpoint=tf.train.get_checkpoint_state('/')
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(test_obj.sess,checkpoint.model_checkpoint_path)

test_accuracy = accuracy.eval(feed_dict={test_obj.x:mnist.test.images,
                              test_obj.y_true:mnist.test.labels})

print "Accuracy on test set:", test_accuracy
