#!/usr/bin/env python3.7
"""
The module of the model or eurekaRes
"""

# import system modules
import os
import sys
import time

# import tensorflow for ANN interface
import tensorflow as tf

# import other helping modules
import json
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import ast

# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU


def strToList(l_str):
    """
    The function converts a string into a list
    with whitespace as separator and returns a list of type np.float64
    """
    # print(l_str)
    l_str = l_str.split(' ')
    # print(l_str)
    l_str = list(map(np.float64, l_str))
    # print(type(l_str))

    return np.array(l_str)


# import the images from csv file (on proper tf format)

# define the dataset for training
ds_name = 'val'

# define the folder that contains the data
train_folder_name = ds_name + '2017'

data_dir = os.environ["PY_WS"]+"/object_detection/" + "COCO_dataset/" + train_folder_name

csv_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/datasets/" + ds_name + "_dataset_test.csv"


data = pd.read_csv(csv_path, usecols=['filename', 'class_id'])

filenames = data.get('filename')
class_labels = data.get('class_id')
nb_classes = len(class_labels.unique())
print("number of classes: ", nb_classes)

data_labels = pd.read_csv(csv_path, usecols=['class_label'], converters={"class_label":strToList})


def process_data(fName, label):
    """
    a function to read and mold the input image and return the image with the corresponding label
    """

    # get the image file
    img = tf.io.read_file(fName)

    # load the image
    img = tf.image.decode_jpeg(img, channels=3)

    # mold the image
    img = tf.image.resize_with_pad(img, 720, 1280)

    # return the image and its correspoding label
    return img, label

# start = time.time()
# im, ll = process_data(tt, data_dir)
# end = time.time()
# print(ll)
# print("time elapsed: ", end-start)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# list_ds = tf.data.Dataset.list_files(data_dir+'*/*')

fileNames = data.pop('filename')
fileNames = data_dir + "/" + fileNames
labels = data_labels.pop('class_label')


list_ds = tf.data.Dataset.from_tensor_slices((fileNames.values, labels))

for fname, clabel in list_ds.take(2):
    print(fname)
    print(clabel)

labeled_ds = list_ds.map(process_data, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

def prepare_for_training(ds, cache=True, shuffle_buffer_size=100):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(100)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(5, 5))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n].astype(np.uint8))
        plt.title("test")
        plt.axis('off')
    
    plt.show()


show_batch(image_batch.numpy(), label_batch.numpy())



# properly set the ANN to accept the images as tensors and the correspoding labels

# finalize ANN architecture

# train ANN

# test ANN on real-time and measure its performance
