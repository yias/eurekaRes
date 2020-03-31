#!/usr/bin/env python3.7

# import numpy as np
import cv2

import eurekaRes_utils

import time

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import pathlib

# Root directory of the project
ROOT_DIR = os.path.abspath(str(pathlib.Path().absolute())+"/Mask_RCNN/")

# Import Mask RCNN
sys.path.insert(0, ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
sys.path.append(str(pathlib.Path().absolute()) + "/coco/PythonAPI")
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']





# create object to capture the frames from an input
cap = cv2.VideoCapture(2)

# set the resolution of the frame
cap.set(3, 1280)
cap.set(4, 720)

# Define the codec and create VideoWriter object
fourCC = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('data/output.avi', fourCC, 30.0, (1280, 720))

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

Colors = eurekaRes_utils.random_colors(50)

timings = np.array([], dtype=np.float64).reshape(0, 1)
start_time = time.time()
frame_counter = 0.0
all_time_start = time.time()

while True:
    try:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # write the flipped frame
        # out.write(frame)

        

        # Run detection
        results = model.detect([frame], verbose=1)

        # # Visualize results
        r = results[0]
        
        
        frame = eurekaRes_utils.draw_boxes(frame, r['rois'], color_pallete=Colors)

        frame = eurekaRes_utils.add_classes_names_to_image(frame, r['rois'], r['class_ids'], class_names, r['scores'], text_colors=Colors)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        timings = np.vstack((timings, time.time()-start_time))
        frame_counter += 1.0
        start_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    except KeyboardInterrupt:
        break

duration = time.time() - all_time_start

# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()

print("fps: ", frame_counter/duration)
print(timings)
print("average process time: ", np.mean(timings))
print("std process time: ", np.std(timings))

