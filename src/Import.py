import tensorflow as tf
import PIL as pl
from PIL import Image as img
import webbrowser
import numpy as np
from scipy import misc
import random


def read_labeled_image_list(path):
    input_image_dir='/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/JPEGImages/'
    input_label_dir='/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/SegmentationClass/'
    f1 = open(path, 'r')
    filenames = []
    labels = []
    for line in f1:
        filenames.append(input_image_dir+line[:-1])
        labels.append(input_label_dir+line[:-1])
    return filenames, labels

def next(num):
    image_list, label_list = read_labeled_image_list('/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')

    from random import randint
    random_ind = randint(0,1000)
    #print random_ind
    image = img.open(image_list[num]+".jpg").resize((256,256) ,img.ANTIALIAS)
    label = img.open(label_list[num]+".png").resize((256,256) ,img.ANTIALIAS)
    #webbrowser.open(image_list[num]+".jpg")
    #webbrowser.open(label_list[num]+".png")
    #label.save("shortened.png")

    if num == 49:
        #webbrowser.open("shortened.png")
        webbrowser.open(label_list[num]+".png")

    webbrowser.open(label_list[num]+".png")


    label_arr = np.array(label)
    image_arr = np.array(image)


    label_arr_new = np.zeros([label_arr.shape[0],label_arr.shape[1]])

    for i in range (255):
        for j in range (255):
            if label_arr[i,j] == 255:
                label_arr_new[i,j] = 20

    #print np.max(label_arr_new)

    label_arr = label_arr_new.astype(np.uint64)
    image_arr = image_arr[np.newaxis, ...]
    label_arr = label_arr[np.newaxis, ...]
    #print label_arr.shape

    return image_arr,label_arr
