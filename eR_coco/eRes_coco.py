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
import time

# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU

# Import COCO utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/coco/PythonAPI")
from pycocotools.coco import COCO

 


# define root directory
dataDir = os.environ["PY_WS"]+"/object_detection/COCO_dataset"

# define the folder that contains the data
dataType = 'train2017'

# define the file that contains the COCO annotations
train_annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# define the file that contains the COCO labels
config_file = "training_info.json"

# initialize COCO api for instance annotations
coco = COCO(train_annFile)

# load labels from json file, the images with these labels will be used for object detection (training and testing)
with open(os.environ["PY_WS"]+"/object_detection/eurekaRes/eR_coco/"+config_file, 'r') as f:
    config_json = json.load(f)

train_labels = config_json.get("labels")

# get all images containing given categories, select one at random
train_catIds = coco.getCatIds(catNms=train_labels)
train_imgIds = coco.getImgIds(catIds=train_catIds[0])

print("number of training images", len(train_imgIds))
timings = np.array([]).reshape(0, 1)

for nb_img in range(0, 10):
    # load instance annotations
    train_annIds = coco.getAnnIds(imgIds=train_imgIds[nb_img], catIds=train_catIds, iscrowd=None)
    train_anns = coco.loadAnns(train_annIds)


    # read image from file
    I = io.imread('%s/%s/%s.jpg'%(dataDir, dataType, str(train_imgIds[nb_img]).zfill(12)))
    print("image file name: ", str(train_imgIds[nb_img]).zfill(12))

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig, ax1 = plt.subplots(1)
    ax1.imshow(I)
    ax1.axis('off')

    train_anns_label = [x for x in train_anns if x.get("category_id") == 1 and x.get("iscrowd") == 0]

    # tt_box = train_anns_label[0].get("bbox")
    # rect = patches.Rectangle(tt_box[:2], tt_box[2], tt_box[3], linewidth=1, edgecolor='r', facecolor='none')
    # ax1.add_patch(rect)

    # For each instance in the picture, create a rectangle patch and add it to the axes
    for inst_box in train_anns_label:
        tt_box = inst_box.get("bbox")
        rect = patches.Rectangle(tt_box[:2], tt_box[2], tt_box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)


    # create an numpy ndarray with all the boxes of the category (i.e. class)
    img_cat_boxes = np.array([np.array(x.get("bbox")) for x in train_anns_label])

    # mold the images to be in the preferred dimensions
    start_t = time.time()
    new_img, new_boxes, window, scale, padding = eResU.mold_image(I, img_cat_boxes,\
        config_json.get("max_height"), config_json.get("max_width"), config_json.get("do_padding"))

    end_t = time.time()
    timings = np.vstack([timings, end_t - start_t])
    # display the molded figure with all the boxes
    ax2.imshow(new_img)
    ax2.axis('off')
    for inst_box in new_boxes:
        rect = patches.Rectangle(inst_box[:2], inst_box[2], inst_box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    fig.suptitle("Image " + str(nb_img), fontsize=16)
    plt.show()

print("average comp time = ", np.mean(timings))
print("std = ", np.std(timings))
# clear coco variable for saving memory
coco = []
