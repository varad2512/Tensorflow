import tensorflow as tf
import PIL as pl
from PIL import Image as img
import webbrowser
import numpy as np
from scipy import misc
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
    import random
    #print image_list[:2]
    #print label_list[:2]

    random_ind = random.randint(1,1000)
    image = img.open(image_list[30]+".jpg")#.resize((256,256))
    image.show()

    image = misc.imread(image_list[3]+".jpg")


    label = img.open(label_list[3]+".png").resize((256,256))

    #has_alpha = label.mode == 'RGB'
    #print has_alpha
    print label
    label_arr = np.array(label).astype(np.float32)
    print label_arr.shape
    '''
    label.load()

    label = label.split()[3]
    label = label.convert('RGB')

    print label_arr
    #label.save(label_list[31]+".jpg",transparency = 0)
    #webbrowser.open(label_list[31]+".png")
    #print label
    #label = tf.image.decode_png(label_list[31], channels = None  , dtype = None, name=None)
    #print label
    '''
    '''
    label = label.convert('RGBA')

    label.load()
    non_transparent=img.new('RGB',label.size,(255,255,255))
    non_transparent.paste(label,mask=label.split()[3])
    label = non_transparent
    #label = misc.imread(label_list[30]+".png",'P')

    print image.shape
    print label
    '''
    #image_arr = np.array(image).astype(np.float32)

    '''
    #print label_arr.dtype
    pal = label.getpalette()
    num_colours = len(pal)/3
    print num_colours
    max_val = float(np.iinfo(label_arr.dtype).max)
    print max_val
    map = np.array(pal).reshape(num_colours, 3) / max_val

    print map.dtype
    print map

    print map[label_arr]
    '''
    '''
    img1 = img.fromarray(map,"RGB")

    img1.save('my.png')

    #print image_arr.shape
    #print label_arr.shape
    '''



    '''
    label_arr = np.dstack(label_arr)
    print label_arr.shape

    label_arr= np.rollaxis(label_arr,-1)
    print label_arr.shape

    label_arr=np.rollaxis(label_arr,-1)
    print label_arr.shape

    image_batch=np.array([image_arr])
    label_batch=np.array([label_arr])


    print label_batch.shape
    print image_batch.shape
    label_batch = np.dstack(label_batch)
    print label_batch.shape

    image_arr = tf.convert_to_tensor(image_batch ,dtype = tf.float32)
    label_arr = tf.convert_to_tensor(label_batch , dtype = tf.int32)

    image_arr = tf.image.resize_images(image_arr, 256, 256, method = 2)
    label_arr = tf.image.resize_images(label_arr, 256, 256, method = 2)

    sess = tf.InteractiveSession()
    label_arr = label_arr.eval()

    label_arr= np.rollaxis(label_arr,-1)
    label_arr = tf.convert_to_tensor(label_arr , dtype = tf.int32)


    #label_arr = tf.reshape(label_arr,[1,256,256])

    return image_arr, label_arr

    '''
next()
