import tensorflow as tf
import PIL as pl
from PIL import Image as img
import webbrowser
import numpy as np
def read_labeled_image_list(path):
    """Reads a .txt file containing paths and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    input_image_dir='/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/JPEGImages/'
    input_label_dir='/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/SegmentationClass/'


    f1 = open(path, 'r')

    filenames = []
    labels = []
    for line in f1:
        filenames.append(input_image_dir+line[:-1])
        labels.append(input_label_dir+line[:-1])
    return filenames, labels
'''
def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label_contents = tf.read_file(input_queue[1])
    file_contents  = tf.read_file(input_queue[0])
    example        = tf.image.decode_jpeg(file_contents, channels=3)
    label          = tf.image.decode_png(label_contents,channels=3)
    return example, label
'''

def next():
    image_list, label_list = read_labeled_image_list('/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
    '''
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.string)
    num_epochs = 1

    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=False)
    image, label = read_images_from_disk(input_queue)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch)

    '''
    #print image_list[:2]
    #print label_list[:2]
    image = img.open(image_list[0]+".jpg")
    label = img.open(label_list[0]+".png")
    #print image
    image_arr = np.array(image).astype(np.float32)
    label_arr = np.array(label).astype(np.int32)
<<<<<<< HEAD
    #print image_arr.shape

    label_arr = np.dstack(label_arr)
    print label_arr.shape
    label_arr= np.rollaxis(label_arr,-1)
    label_arr=np.rollaxis(label_arr,-1)
    print label_arr.shape

    image_batch=np.array([image_arr])
    label_batch=np.array([label_arr])


    #print label_arr.shape

    image_arr = tf.convert_to_tensor(image_batch ,dtype = tf.float32)
    label_arr = tf.convert_to_tensor(label_batch , dtype = tf.int32)
    image_arr = tf.image.resize_images(image_arr, 256, 256, method = 2)
    label_arr = tf.image.resize_images(label_arr, 256, 256, method = 2)

    return image_arr, label_arr
#next()
=======

    

    image_batch=np.array([image_arr])
    label_batch=np.array([label_arr])

    print image_batch.shape
    print label_batch.shape

#    return image_batch,label_batch
next()
>>>>>>> ee3267f7e0705347cb9f4e75bf1933f7de4cf5a8
