#!/usr/bin/env python3.7
"""
The module for predicting with eurekaRes
"""

# import system modules
import os
import sys


import time
from datetime import datetime
import json
import random

# import tensorflow for ANN interface
import tensorflow as tf

# import other helping modules
import numpy as np
# import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


# path to saved model
save_path = os.environ["PY_WS"] + "/object_detection/eurekaRes/" + "data/models/" + "2020/04/14_1534"

# read the config file
config_file = "coco_config.json"
with open(os.environ["PY_WS"] + "/object_detection/eurekaRes/eR_coco/" + config_file, 'r') as f:
    config_params = json.load(f)

# get the desired image
image_shape = (config_params.get("max_height"), config_params.get("max_width"))

# get the labels
labels = config_params.get('labels').get('objects')

# load the validation csv file
ds_name = "val"
train_folder_name = ds_name + '2017'

# define the path of the csv file and read its contents
csv_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/datasets/" + ds_name + "_dataset_test.csv"

# get the filenames of the images and the corresponding class ids
data = pd.read_csv(csv_path, usecols=['filename', 'class_id'])


# isolate the collumn with the file names of the images
fileNames = data.pop('filename')

# define the folder containing the data
data_dir = os.environ["PY_WS"]+"/object_detection/" + "COCO_dataset/" + train_folder_name


# get a random image
img_fname = fileNames[random.randrange(len(fileNames))]
img_fname = data_dir + "/" + img_fname


# load the image
img = tf.io.read_file(img_fname)
img = tf.image.decode_jpeg(img, channels=3)

# mold the image
img = tf.image.resize_with_pad(img, image_shape[0], image_shape[1])
print(type(img))
img = img/255.0
print(type(img))
img = img.numpy().reshape([-1, image_shape[0], image_shape[1], 3])

# load a trained model
print("loading trained model ... ")
trained_model = tf.keras.models.load_model(save_path)

print(img.numpy().shape)
print(image_shape)

redicted_label = trained_model.predict([img])

print(type(redicted_label))
print(redicted_label)


# fig, ax = plt.subplots(1, figsize=(12, 9))
# ax.axis("off")
# ax.imshow(img.numpy().astype('uint8'))
# plt.show()
