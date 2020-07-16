#!/usr/bin/env python3.7

# import numpy as np
import cv2

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import pathlib
import time

# Import gaze_tracking module
sys.path.append(str(pathlib.Path().absolute()) + "/../Aruco")
from ArucoBoardDetection import CameraToWorld
# exit()



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configgpu = ConfigProto()
configgpu.gpu_options.allow_growth = True
session = InteractiveSession(config=configgpu)


# import eurekaRes utilities
if sys.platform == 'linux':
    sys.path.append(str(pathlib.Path().absolute()) + "/../utils")
else:
    sys.path.append(str(pathlib.Path().absolute()) + "\\..\\utils")
import eurekaRes_utils


# Root directory of the project
ROOT_DIR = os.path.abspath((os.environ["PY_WS"]+"/Mask_RCNN/"))

# Import Mask RCNN
sys.path.insert(0, ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
sys.path.append(os.environ["PY_WS"] + "/coco/PythonAPI")
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




dataFolder = str(pathlib.Path().absolute()) + "/../data/"
vFileName = "gazeRecordings/aruco/recording_aruco_gaze_world_clean.avi"
outVFileName = "gazeRecordings/aruco/gaze_objects_pos_t05.mp4"
gazeFName = "gazeRecordings/aruco/recording_aruco_gaze.csv"
csvOutputFile = "gazeRecordings/gaze_objects_pos.csv"

area_threshold = 25000
clf_threshold = 0.6
frame_history_length = 2
area_around_cm = 100
bboxes_history = []
cm_history = []
prediction_history = []
score_history = []
valid_frame_counter = 0

coord_df = pd.read_csv(dataFolder + gazeFName)

# print(coord_df.shape)

gaze_coord = coord_df[['gaze_x', 'gaze_y']].fillna(-1)
gaze_coord = gaze_coord.to_numpy()

# print(type(gaze_coord))
# print(gaze_coord.shape)
# print(gaze_coord[9, :])

# if not np.isnan(gaze_coord[9][0]):
#     print("is nan")
for i in range(gaze_coord.shape[0]):
    if isinstance(gaze_coord[i, 0], str):
        gaze_coord[i, 0] = float(gaze_coord[i, 0][1:-1])
        gaze_coord[i, 1] = float(gaze_coord[i, 1][1:-1])
    # # print(len(gaze_coord[int(i), 0]))
    # if gaze_coord[i, 0] < 0:
    #     print(i)


# exit()

# create object to capture the frames from an input
cap = cv2.VideoCapture(dataFolder + vFileName)
# print(dataFolder + vFileName)

# set the resolution of the frame
cap.set(3, 1280)
cap.set(4, 720)

# Define the codec and create VideoWriter object
fourCC = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(dataFolder + outVFileName, fourCC, 10.0, (1280, 720))

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

Colors = eurekaRes_utils.random_colors(50)

timings = np.array([], dtype=np.float64).reshape(0, 1)
start_time = time.time()
frame_counter = 0.0

ar = []
all_time_start = time.time()

print('test before staring')
# exit()

while cap.isOpened():
    # print('test')
    try:

        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break

        # Run detection
        # results = model.detect([frame], verbose=0)

        # # # Visualize results
        # r = results[0]
        # bboxes = r['rois']
        # scores = r['scores']
        # bboxes = bboxes[scores > clf_threshold, :]

        # predicted_labels = []
        # tt = r['class_ids']
        # for ll in tt:
        #     predicted_labels += [class_names[ll]]

        # predicted_labels = np.array(predicted_labels)
        # predicted_labels = predicted_labels[scores > clf_threshold]
        # scores = scores[scores > clf_threshold]
        
        # cm_bxs, size_bxs, bx_area = eurekaRes_utils.get_cm(bboxes)
        # bboxes = bboxes[bx_area < area_threshold, :]
        # predicted_labels = predicted_labels[bx_area < area_threshold]
        # cm_bxs = cm_bxs[bx_area < area_threshold, :]
        # # print("cm_bxs.shape: ", cm_bxs.shape)
        # size_bxs = size_bxs[bx_area < area_threshold, :]
        

        # if valid_frame_counter < frame_history_length:
        #     if bboxes.any():
        #         bboxes_history += [bboxes]
        #         cm_history += [cm_bxs]
        #         prediction_history += [predicted_labels]
        #         score_history += [scores]
        #         valid_frame_counter += 1
        # else:
        #     if bboxes.any():
        #         bboxes_history += [bboxes]
        #         cm_history += [cm_bxs]
        #         prediction_history += [predicted_labels]
        #         score_history += [scores]
        #         bboxes, cm_bxs, predicted_labels, scores = eurekaRes_utils.average_prediction(bboxes_history, cm_history, prediction_history, score_history, aacm=area_around_cm, clThr=clf_threshold)
        #         del bboxes_history[0]
        #         del prediction_history[0]
        #         del cm_history[0]
        #         del score_history[0]

        #     realWorld_coord, rwc_check = CameraToWorld(cm_bxs, frame)
        #     print("real world coord: ", realWorld_coord)
        #     rbg_clr = np.array([], dtype=int).reshape(0, 3)
        #     cc = 0
        #     for rro in bboxes:
        #         if gaze_coord[int(frame_counter), 0] > 0:
        #             # print(gaze_coord[int(frame_counter), :], rro, predicted_labels[cc]) 
        #             if (gaze_coord[int(frame_counter), 0] >= rro[1]) and (gaze_coord[int(frame_counter), 0] <= rro[3]):
        #                 if (gaze_coord[int(frame_counter), 1] >= rro[0]) and (gaze_coord[int(frame_counter), 1] <= rro[2]):
        #                     rbg_clr = np.vstack((rbg_clr, np.array([0, 0, 255])))
        #                 else:
        #                     rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))
        #             else:
        #                 rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))
        #         else:
        #             rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))
        #         cc += 1

            
            # frame = eurekaRes_utils.draw_boxes(frame, bboxes, color_pallete=rbg_clr)
            
            # frame = eurekaRes_utils.add_classes_names_to_image(frame, bboxes, predicted_labels, scores, text_colors=rbg_clr)
            
            # if rwc_check:
            #     frame = eurekaRes_utils.display_real_coord(frame, bboxes, realWorld_coord, text_colors=rbg_clr)
                # for kk in cm_bxs:
                #    cv2.circle(frame, (int(kk[0]), int(kk[1])), 10, (0, 255, 0), -5)

        # if gaze_coord[int(frame_counter), 0] > 0:
        #     # print(gaze_coord[int(frame_counter), 0], gaze_coord[int(frame_counter), 1])
        #     gzc = (int(gaze_coord[int(frame_counter), 0]), int(gaze_coord[int(frame_counter), 1]))
        #     # print(gzc)
        #     cv2.circle(frame, (gzc), 20, (0, 0, 255), -5)

        # # Display the resulting frame
        # # cv2.imshow('frame', frame)
        # timings = np.vstack((timings, time.time()-start_time))
        # ar += [[gaze_coord[int(frame_counter), 0], gaze_coord[int(frame_counter), 1], str(predicted_labels), str(bboxes), str(scores[scores > clf_threshold])]] # 
        # frame_counter += 1.0
        # print(frame_counter)
        # # write the flipped frame
        # out.write(frame)
        # start_time = time.time()
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        # visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    except KeyboardInterrupt:
        break

duration = time.time() - all_time_start

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

header = ['gaze_x', 'gaze_y', 'predicted_labels', 'boxes', 'scores']
# print(ar)
# print(len(ar))
# print(len(ar[0]))
drame = pd.DataFrame(ar, columns=header)
drame.to_csv(dataFolder + csvOutputFile)

print("fps: ", frame_counter/duration)
# print(timings)
print("average process time: ", np.mean(timings))
print("std process time: ", np.std(timings))

