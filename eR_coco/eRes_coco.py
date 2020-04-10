#!/usr/bin/env python3.7

import os
import sys

# import tensorflow for creating the datasets
# import tensorflow as tf

# import other helping modules
import json
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import time

# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU

# Import COCO utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/coco/PythonAPI")
from pycocotools.coco import COCO


# define if train or validation data
ds_name = 'train'

# define categories of interest
cats_of_interest = ['person', 'objects']

# set the plot to appear or not
show_plot = False



# define the folder that contains the data
dataType = ds_name + '2017'


# define root directory
dataDir = os.environ["PY_WS"]+"/object_detection/COCO_dataset"

# define the file that contains the COCO annotations
train_annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# define the file that contains the COCO labels
config_file = "coco_config.json"


# define the output path and file name
output_file_name = ds_name + '_dataset.csv'
output_save_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/" + "data/" + output_file_name

# initialize COCO api for instance annotations
coco = COCO(train_annFile)

# load labels from json file, the images with these labels will be used for object detection (training and testing)
with open(os.environ["PY_WS"]+"/object_detection/eurekaRes/eR_coco/"+config_file, 'r') as f:
    config_json = json.load(f)

# load all the labels
label_super_cats = config_json.get("labels")

train_labels = []
for cOI in cats_of_interest:
    train_labels = train_labels + label_super_cats.get(cOI)
# train_labels = [label_super_cats.get(cOI) for cOI in cats_of_interest] TODO: nested format of the for loop


print("number of classes: ", len(train_labels))
print("train_labels: ", train_labels)

nb_classes = len(train_labels)

# get all images containing given categories, select one at random
train_catIds = coco.getCatIds(catNms=train_labels)
# train_imgIds = coco.getImgIds(catIds=train_catIds[0])

print("number of catIDs", len(train_catIds))
# print("number of training images", len(train_imgIds))
timings = np.array([]).reshape(0, 1)

label_format = np.zeros(nb_classes)
label_index = 0
df_rows = []

for catId in train_catIds:
    train_imgIds = coco.getImgIds(catIds=catId)
    print("[class: " + train_labels[label_index] + ", COCO id :" + str(catId) + "] number of images: " + str(len(train_imgIds)))

    # define the label of the class
    class_label = np.copy(label_format)
    class_label[label_index] = 1.0
    label_index += 1

    for nb_img in range(len(train_imgIds)):
        # load instance annotations
        train_annIds = coco.getAnnIds(imgIds=train_imgIds[nb_img], catIds=train_catIds, iscrowd=None)
        train_anns = coco.loadAnns(train_annIds)

        # create the path of the image
        file_path = dataDir + "/" + dataType + "/"
        file_name = str(train_imgIds[nb_img]).zfill(12) + ".jpg"
        print("image file name: ", file_name)

        # read image from file
        I = io.imread(file_path+file_name)

        train_anns_label = [x for x in train_anns if x.get("category_id") == catId and x.get("iscrowd") == 0]

        # create an numpy ndarray with all the boxes of the category (i.e. class)
        img_cat_boxes = np.array([np.array(x.get("bbox")) for x in train_anns_label])

        # mold the images to be in the preferred dimensions
        start_t = time.time()
        new_img, new_boxes, window, scale, padding = eResU.preprocess_image(I, img_cat_boxes,\
            config_json.get("max_height"), config_json.get("max_width"), config_json.get("do_padding"))

        end_t = time.time()
        timings = np.vstack([timings, end_t - start_t])

        for inst_box in new_boxes:
            df_rows = df_rows + [[file_name, str(class_label)] + list(map(str, inst_box))]

        if show_plot:
            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
            # fig, ax1 = plt.subplots(1)
            ax1.imshow(I)
            ax1.axis('off')
            ax1.set_title('original image with annotations')
            # For each instance in the picture, create a rectangle patch and add it to the axes
            for inst_box in train_anns_label:
                tt_box = inst_box.get("bbox")
                rect = patches.Rectangle(tt_box[:2], tt_box[2], tt_box[3], linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)

            # display the molded figure with all the boxes
            ax2.imshow(new_img)
            ax2.axis('off')
            ax2.set_title('molded image with annotations')

            for inst_box in new_boxes:
                # df_rows = df_rows + [[file_name, str(class_label)] + list(map(str, inst_box))]
                rect = patches.Rectangle(inst_box[:2], inst_box[2], inst_box[3], linewidth=1, edgecolor='r', facecolor='none')
                ax2.add_patch(rect)

            fig.suptitle("class: " + train_labels[label_index - 1] + "\nImage " + str(nb_img), fontsize=16)
            plt.show()

# clear coco variable for saving memory
coco = []
df = pd.DataFrame(df_rows, columns=["filename", "class", "xmin", "ymin", "xmax", "ymax"])

df.to_csv(output_save_path)

print("output file generated in: ", output_save_path)

print("average comp time = ", np.mean(timings))
print("std = ", np.std(timings))
