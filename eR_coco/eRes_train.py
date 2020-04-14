#!/usr/bin/env python3.7
"""
The module of the model of eurekaRes
"""

# import system modules
import os
import sys
import time
from datetime import datetime
import json

# import tensorflow for ANN interface
import tensorflow as tf

# import other helping modules
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
# import ast

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
    l_str = list(map(np.float32, l_str))
    # print(type(l_str))

    return np.array(l_str)

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(5, 5))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n].astype(np.uint8))
        plt.title("test")
        plt.axis('off')

    plt.show()


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

    # normalize image
    # img = tf.image.per_image_standardization(img)
    img = img/255

    # return the image and its correspoding label
    return img, label



def prepare_ds(ds, batchSize, cache=False, shuffle_buffer_size=10, autoTune=tf.data.experimental.AUTOTUNE):
    """
    The function
    """

    # enable the load to cache if the dataset is small (<1GB)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # randomly shuffle the elements of the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    # Repeat forever
    ds = ds.repeat()

    # set the batchsize
    ds = ds.batch(batchSize)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.

    # ds = ds.prefetch(buffer_size=autoTune)

    return ds

def create_ds(ds_name, batchSize):
    """
    The function get the indicator of the nane of the csv file
    and returns the corresponding dataset
    """

    # define the folder that contains the data
    train_folder_name = ds_name + '2017'

    data_dir = os.environ["PY_WS"]+"/object_detection/" + "COCO_dataset/" + train_folder_name

    # define the path of the csv file and read its contents
    csv_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/datasets/" + ds_name + "_dataset_test.csv"

    # get the filenames of the images and the corresponding class ids
    data = pd.read_csv(csv_path, usecols=['filename', 'class_id'])

    # get the classes ids from the data and print the number of classes of the dataset
    class_labels = data.get('class_id')
    nb_classes = len(class_labels.unique())
    print("Creating " + ds_name + " set ...")
    print("number of classes: ", nb_classes)
    print("number of samples: ", len(data))

    # get the corresponing labels of the images, and convert them to list
    data_labels = pd.read_csv(csv_path, usecols=['class_label'], converters={"class_label":strToList})

    # prepare the dataset

    # isolate the collumn with the file names of the images
    fileNames = data.pop('filename')

    # add the path of the folder of the images
    fileNames = data_dir + "/" + fileNames

    # get the labels of the images
    labels = data_labels.pop('class_label')
    # labels = tf.keras.utils.to_categorical(labels)

    # create a tf dataset with the filenames of the images and their labels
    list_ds = tf.data.Dataset.from_tensor_slices((fileNames.values, labels))

    AUTOTUNE = tf.data.experimental.AUTOTUNE  #TODO: make AUTOTUNE class member variable

    # apply a tranformation to the dataset, for converting the filenames to images
    # and mold the images
    labeled_ds = list_ds.map(process_data, num_parallel_calls=AUTOTUNE) #

    # check if the transformation is correct by printint the first tensor
    # for image, img_label in labeled_ds.take(1):
    #     print("Image shape: ", image.numpy().shape, image.numpy().dtype)
    #     print(np.amax(image.numpy()), np.amin(image.numpy()))
    #     print("Label: ", img_label.numpy(), img_label.numpy().shape, img_label.numpy().dtype)

    # set the preferred option to the dataset
    r_ds = prepare_ds(labeled_ds, batchSize, autoTune=AUTOTUNE)

    print("-----------------------------------")

    return r_ds, nb_classes


def create_model(nb_classes, img_shape):
    """
    ANN architecture
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(nb_classes+4, activation='softmax'))

    return model



#############################################################################################
#############################################################################################
#############################################################################################




# load the parameters from a config file
config_file = "coco_config.json"
with open(os.environ["PY_WS"] + "/object_detection/eurekaRes/eR_coco/" + config_file, 'r') as f:
    config_params = json.load(f)

# get the desired image
image_shape = (config_params.get("max_height"), config_params.get("max_width"))

# get the batch size
BATCHSIZE = config_params.get("BATCH_SIZE")

# get the number of epochs for training
NB_EPOCHS = config_params.get("NB_EPOCHS")


# path to save the model
save_path = os.environ["PY_WS"] + "/object_detection/eurekaRes/" + "data/models/"

# name of the saved model based on the current date and time
now = datetime.now() # current date and time
model_save_name = now.strftime("%Y_%m_%d_%H%M")


# import the datasets from the csv files and create datasets on proper tf format

# define the validation dataset
ds_name_t = 'val'

# create the validation dataset
val_ds, val_nb_classes = create_ds(ds_name_t, BATCHSIZE)

# define the training dataset
ds_name_t = 'train'

# create the validation dataset
train_ds, train_nb_classes = create_ds(ds_name_t, BATCHSIZE)

# if val_nb_classes != train_nb_classes:
#     print("the number of classes of the training set do not match the number of classes of the validation set")

# activate the following lines to inspect the images in the batch
# image_batch, label_batch = next(iter(train_ds))
# show_batch(image_batch.numpy()*255, label_batch.numpy())


# create a model
eRes_model = create_model(train_nb_classes, image_shape)

# print the model summary
eRes_model.summary()


# train the model

eRes_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print("Start training ...")
start = time.time()
eRes_model.fit(train_ds, epochs=NB_EPOCHS, steps_per_epoch=20)
end = time.time()
print("Training complete")
print("Training time: ", end - start)

# evaluate the model
print("Evaluating the model ...")

train_loss, train_accuracy = eRes_model.evaluate(train_ds, steps=30)
print("On the training set:")
print("Loss :", train_loss)
print("Accuracy :", train_accuracy)

val_loss, val_accuracy = eRes_model.evaluate(val_ds, steps=30)
print("On the validation set:")
print("Loss :", val_loss)
print("Accuracy :", val_accuracy)


# save the model
eRes_model.save(save_path + model_save_name)

# test ANN on real-time and measure its performance
