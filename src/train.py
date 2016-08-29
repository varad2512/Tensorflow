 #import ImgSeg
import tensorflow as tf
import FCN_32
from Import import *
import numpy as np
import webbrowser
from PIL import Image as img

train_obj = FCN_32.FCN()
train_obj.graph_build()

print "Training..."
#h_FC3_1            = tf.image.resize_bilinear(train_obj.h_FC3, [256,256], align_corners=None, name=None)
reshaped_logits    = tf.reshape(train_obj.transpose_conv, [-1, 22])  # shape [batch_size*256*256, 20]
reshaped_labels    = tf.reshape(train_obj.y_true, [-1])              # shape [batch_size*256*256]

loss               = tf.nn.sparse_softmax_cross_entropy_with_logits(reshaped_logits,
                                                     reshaped_labels,name="cost_tensor")

loss               = tf.reshape(loss,[1,256*256])
loss               = tf.reduce_sum(loss,1)
loss               = tf.reduce_mean(loss)

train_step         = tf.train.AdamOptimizer(1e-4).minimize(loss)
#train_step         = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#softmax            = tf.nn.softmax(reshaped_logits)
#softmax            = tf.reshape(softmax , [1,256,256,21])


temp               = tf.argmax(train_obj.transpose_conv,3)
'''
temp               = tf.reshape(temp,[1,tf.shape(train_obj.transpose_conv)[1],tf.shape(train_obj.transpose_conv)[2]  ])
correct_prediction = tf.equal(tf.reshape(temp,[-1]),tf.reshape(train_obj.y_true,[-1]))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''



'''
l_cross_entropy    = tf.reduce_mean(-tf.reduce_sum(train_obj.y_true
                     * tf.log(train_obj.y_pred), reduction_indices=[1]))
train_step         = tf.train.AdamOptimizer(1e-4).minimize(l_cross_entropy)
'''
'''
correct_prediction = tf.equal(tf.argmax(train_obj.y_pred,1),
                     tf.argmax(train_obj.y_true,1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''


#tf.scalar_summary("cost_train",loss)
#summary_op         = tf.merge_all_summaries()

summary_tensor     = [["cost_tensor", loss]]#["accuracy_train", accuracy]#]
summary_op         = train_obj.summary_write(summary_tensor)
writer             = tf.train.SummaryWriter("/home/varad/work/Tensorflow/Image_Segmentation/Tensorboard",
                     train_obj.sess.graph)

#tf.scalar("loss",loss)

'''
variable_list_save = [train_obj.w_conv1,train_obj.b_conv1,train_obj.w_conv2
                     ,train_obj.b_conv2,train_obj.w_fc1,train_obj.b_fc1,
                      train_obj.w_fc2,train_obj.b_fc2]
saver              =  tf.train.Saver(variable_list_save)
'''
train_obj.sess.run(tf.initialize_all_variables())

for i in range(100):
    image,label = next(i)

    temp_2,temp_1,summary_accuracy,step = train_obj.sess.run([train_obj.transpose_conv,temp,summary_op,train_step],
                                feed_dict = {train_obj.x:image,
                               train_obj.y_true:label})
    print i
    print np.max(temp_1)

    #f i % 50 == 0:
    writer.add_summary(summary_accuracy,i)

#print accu


print "k",temp_2

colour_map = [[ 0, 0 ,0],
              [ 128,0 ,0],
              [0,128,0],
              [128,128,0],
              [0,0,128],
              [128,0,128],
              [0,128,128],
              [128,128,128],
              [64,0,0],
              [192,0,0],
              [64,128,0],
              [192,128,0],
              [64,0,128],
              [192,0,128],
              [64,128,128],
              [192,128,128],
              [0,64,0],
			  [128,64,0],
              [128,192,0],
			  [0,192,0],
              [0,64,128],
              [224,224,192]]

#temp_1= temp_1.reshape(temp_1.shape[1:])
#colour_map = np.asarray(colour_map)
l=np.zeros([256,256])
for i in range (255):
    for j in range (255):
        l[i,j] = temp_1[0,i,j]
l=l.astype(np.uint8)



#temp_1 = colour_map[temp_1]
final = np.zeros([256,256,3])

for i in range (255):
    for j in range (255):
        index = l[i,j]
        final[i,j,0] = colour_map[index][0]
        final[i,j,1] = colour_map[index][1]
        final[i,j,2] = colour_map[index][2]

print final.shape
print final
final = final.astype(np.uint8)

img1 = img.fromarray(final,'RGB')
img1.show()
img1.save("my.png")
#my = img.open("my.png").resize((256,256))
webbrowser.open("my.png")
