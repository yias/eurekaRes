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

from socketStream_py import socketStream

# Import gaze_tracking module
# sys.path.append(str(pathlib.Path().absolute()) + "/../Aruco")
# from ArucoBoardDetection import CameraToWorld
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




# dataFolder = str(pathlib.Path().absolute()) + "/../data/"
# vFileName = "gazeRecordings/aruco/recording_aruco_gaze_world_clean.avi"
# outVFileName = "gazeRecordings/aruco/gaze_objects_pos_t05.mp4"
# gazeFName = "gazeRecordings/aruco/recording_aruco_gaze.csv"
# csvOutputFile = "gazeRecordings/gaze_objects_pos.csv"

# area_threshold = 25000
# clf_threshold = 0.6
# frame_history_length = 2
# area_around_cm = 100
# bboxes_history = []
# cm_history = []
# prediction_history = []
# score_history = []
# valid_frame_counter = 0

# coord_df = pd.read_csv(dataFolder + gazeFName)

# print(coord_df.shape)

# gaze_coord = coord_df[['gaze_x', 'gaze_y']].fillna(-1)
# gaze_coord = gaze_coord.to_numpy()

# print(type(gaze_coord))
# print(gaze_coord.shape)
# print(gaze_coord[9, :])

# if not np.isnan(gaze_coord[9][0]):
#     print("is nan")
# for i in range(gaze_coord.shape[0]):
#     if isinstance(gaze_coord[i, 0], str):
#         gaze_coord[i, 0] = float(gaze_coord[i, 0][1:-1])
#         gaze_coord[i, 1] = float(gaze_coord[i, 1][1:-1])
    # # print(len(gaze_coord[int(i), 0]))
    # if gaze_coord[i, 0] < 0:
    #     print(i)


# exit()

# create object to capture the frames from an input
# cap = cv2.VideoCapture(dataFolder + vFileName)
# print(dataFolder + vFileName)

# set the resolution of the frame
# cap.set(3, 1280)
# cap.set(4, 720)

# Define the codec and create VideoWriter object
# fourCC = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(dataFolder + outVFileName, fourCC, 10.0, (1280, 720))

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Colors = eurekaRes_utils.random_colors(50)

# initialize socketStream
sockHndlr = socketStream.socketStream(svrIP="128.178.145.17", socketStreamMode=1)
sockHndlr.setBufferSize(128)
sockHndlr.initialize_msgStruct(["labels", "bboxes", "scores", "masks"])

everything_ok = False
if sockHndlr.initialize_socketStream() == 0:
    if sockHndlr.runServer() == 0:
        everything_ok = True


timings = np.array([], dtype=np.float64).reshape(0, 1)
start_time = time.time()
frame_counter = 0.0

ar = []
all_time_start = time.time()

print('ready to classify')
# exit()

if everything_ok:
    while True:
        # print('test')
        try:

            # # Capture frame-by-frame
            # ret, frame = cap.read()
            
            # if not ret:
            #     break

            if sockHndlr.socketStream_ok():
                msg=sockHndlr.get_latest()
                if msg is not None:
                    clf_th = msg['clf_threshold']
                    # print(clf_th)
                    img_dims = np.array(msg['img_dims'], dtype=np.int)
                    # print(img_dims)
                    # print(msg['img'])
                    frame = np.array(msg['img'], dtype=np.int32)
                    # print(frame)
                    # print(frame.shape)
                    # print(frame.dtype)
                    frame = frame.reshape(img_dims).astype(np.uint8)
                    # print("frame shape ", frame.shape, frame.dtype)
                    # Run detection
                    results = model.detect([frame], verbose=0)

                    r = results[0]
                    bboxes = r['rois']
                    # print("bboxes ", type(bboxes), bboxes.shape, bboxes.dtype)
                    scores = r['scores']
                    # print("scores ", type(scores), scores.shape, scores.dtype)
                    tt = r['class_ids']
                    # print("tt ", type(tt), tt.shape, tt.dtype)
                    bboxes = bboxes[scores > clf_th, :]
                    tt = tt[scores > clf_th]
                    scores = scores[scores > clf_th]
                    
                    # predicted_labels = []

                    # for ll in tt:
                    #     predicted_labels += [class_names[ll]]

                    # predicted_labels = np.array(predicted_labels)
                    # predicted_labels = predicted_labels[scores > clf_th]
                    
                    # print('bboxes')
                    sockHndlr.updateMSG('bboxes', bboxes)
                    # print('scores')
                    sockHndlr.updateMSG('scores', scores)
                    # print('labels')
                    sockHndlr.updateMSG('labels', tt)
                    sockHndlr.sendMSg2All()


                    timings = np.vstack((timings, time.time()-start_time))
                    
                    frame_counter += 1.0
                    print(frame_counter)
                    # # write the flipped frame
                    # out.write(frame)
                    start_time = time.time()
            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        except KeyboardInterrupt:
            break

sockHndlr.closeCommunication()

duration = time.time() - all_time_start

# When everything done, release the capture
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# header = ['gaze_x', 'gaze_y', 'predicted_labels', 'boxes', 'scores']
# print(ar)
# print(len(ar))
# print(len(ar[0]))
# drame = pd.DataFrame(ar, columns=header)
# drame.to_csv(dataFolder + csvOutputFile)

print("fps: ", frame_counter/duration)
# print(timings)
if timings.any():
    print("average process time: ", np.mean(timings[1:]))
    print("std process time: ", np.std(timings[1:]))

# fig, ax = plt.subplots()

# ax.plot(timings)
# ax.set_title('detection time')
# ax.set_xlabel('frame')
# ax.set_ylabel('time [s]')
# plt.savefig(dataFolder + 'detection_time.png', dpi=300)

# fig2, ax2 = plt.subplots()
# ax2.plot(timings[1:])
# ax2.set_title('detection time')
# ax2.set_xlabel('frame')
# ax2.set_ylabel('time [s]')
# plt.savefig(dataFolder + 'detection_time2.png', dpi=300)
# plt.show()
