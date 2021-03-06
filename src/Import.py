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
    print len(image_list)
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
        webbrowser.open(label_list[1]+".png")

    #webbrowser.open(label_list[num]+".png")


    label_arr = np.asarray(label)
    image_arr = np.asarray(image)


    label_arr_new = np.zeros([256,256])

    for i in range (255):
        for j in range (255):
            if label_arr[i,j] == 255:
                  label_arr_new[i,j] = 21
            else:
                  label_arr_new[i,j] = label_arr[i,j]



    label_arr = label_arr_new.astype(np.uint64)
    image_arr = image_arr[np.newaxis, ...]
    label_arr = label_arr[np.newaxis, ...]


    return image_arr,label_arr


def next_test():
    image_list, label_list = read_labeled_image_list('/home/varad/Desktop/Dataset/VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')
    image = img.open(image_list[0]+".jpg").resize((256,256) ,img.ANTIALIAS)
    webbrowser.open(image_list[0]+".jpg")
    image_arr = np.asarray(image)
    image_arr = image_arr[np.newaxis, ...]
    return image_arr
