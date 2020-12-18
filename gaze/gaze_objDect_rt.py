#!/usr/bin/env python3.7

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
import joblib

import pathlib
import time

# Import gaze_tracking module
sys.path.append(str(pathlib.Path().absolute()) + "/../Aruco")
from ArucoBoardDetection import CameraToWorld
# exit()

# gaze tracker root_dir
# gaze_root = os.path.abspath((os.environ["PY_WS"]+"/gaze_tracking_object_recognition/Gaze_tracking_Object_Detection"))

gaze_root = os.path.abspath((os.environ["PY_WS"]+"/eurekaRes/gaze/gaze_utils/"))

# import gaze utilities
sys.path.append(gaze_root)


# from ArucoMarkerDetection import Worldcoord
from WorldCameraOpening import WorldCameraFrame
from EyeCameraOpening import EyeCameraFrame
import gazeTracking_utils as gazeUt


from socketStream_py import socketStream


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configgpu = ConfigProto()
configgpu.gpu_options.allow_growth = False
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

# Import COCO configuration
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

# confidence threshold (for excluding detected objects of small confidence-score)
clf_threshold = 0.8

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = clf_threshold

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



# set the path and the file names of the output video files
dataFolder = str(pathlib.Path().absolute()) + "/../data/"
outVFileName = "gaze_objects_pos_20201218.mp4"

# threshold for excluding large objects
area_threshold = 40000



# set the number of previous frames used for the averaged prediction (frame_history_length + the current frame)
frame_history_length = 2

# initializing variables regarding the frame history
bboxes_history = []
cm_history = []
prediction_history = []
score_history = []
valid_frame_counter = 0

# load the SVR models for the gaze (one model for each coordinate x,y)
regr_cx = joblib.load('gaze_model/SVR_model_cx.sav')
regr_cy = joblib.load('gaze_model/SVR_model_cy.sav')


# area around the center of mass of the object for identification of the object of interest.
# if gaze is inside this area around the center of mass of a detected object, the object of interest would be identified as this object
area_around_cm = 100


# initialize socketStream for broadcasting the location of the detected objects

# define a socketStream object in a client mode (socketStreamMode=0)
# set the IP (svrIP) and port (svrPort) of the PC that a socketStream server is running
# NB: the socketStream server should be launched first
sockClient = socketStream.socketStream(svrIP="128.178.145.15", socketStreamMode=0, svrPort=10353)

# set the buffer size of the TCP/IP communication
sockClient.setBufferSize(64)

# set the client's name (optional)
sockClient.set_clientName("gaze_client")

# initialize the structure of the message (i.e., the names of the fields of the message)
# our message has 4 fields with names: clf_threshold, obj_location, bboxes, oboi
# clf_threshold: the confidence threshold
# obj_location: the coordinates (in centimeters) of the middle point of the bottom side of the boounding box
#               The coordinates are with respect to the frame of the aruco-board
#               this field contains the coordinates of all the detected objects, except the object of interest (if any)
# bboxes: the bounding boxes (output of the detector) in pixels
# oboi: the coordinates (in centimeters) of the middle point of the bottom side of the bounding box of the object of the interest (identified by the gaze)
#       the coordinates are with respect to the frame of the aruco board
sockClient.initialize_msgStruct(["clf_threshold", "obj_location", "bboxes", "oboi"])

# introduce the value of the confidence threshold to the correspoding field of the message
sockClient.updateMSG("clf_threshold", clf_threshold)

# initialize socketStream and attemp a connection to the socketStream server
everything_ok = False
if sockClient.initialize_socketStream() == 0:
    if sockClient.make_connection() == 0:
        everything_ok = True

if not everything_ok:
    print('No socketStream is running in the given IP. Continue without broadcasting')


# Define the codec and create VideoWriter object
fourCC = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(dataFolder + outVFileName, fourCC, 4.0, (1280, 720))

# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# initialize a vector for holding the duration of each loop (processing time of each loop) for statistical purposes (optional)
timings = np.array([], dtype=np.float64).reshape(0, 1)
start_time = time.time()
frame_counter = 0.0

# ar = []
all_time_start = time.time()

print('test before staring')


while True:
    # print('test')
    try:


        # capture a frame from the world-camera
        frame = WorldCameraFrame()

        # capture the frames from the two cameras looking at the pupils
        cap_left, cap_right = EyeCameraFrame()

        # run detection
        results = model.detect([frame], verbose=0)

        # extract the results (from the )
        r = results[0]

        # get the bounding boxes
        bboxes = r['rois']

        # get the scores
        scores = r['scores']

        # keep only the bounding boxes whose scores are above the confidence threshold
        bboxes = bboxes[scores > clf_threshold, :]

        # get the predicted labels for the objects and convert them to a list
        tt = r['class_ids'] # this returns the ids of the labels of the coco dataset
        predicted_labels = [class_names[ll] for ll in tt]
        predicted_labels = np.array(predicted_labels)

        # keep only the labels whose score is above the confidence threshold
        predicted_labels = predicted_labels[scores > clf_threshold]

        # keep only the scores that are above the confidence threshold
        scores = scores[scores > clf_threshold]

        # get the coordinates, the boxes' lenghts and the area of the predicted objects
        # cm_bxs: the coordinates (in pixels) of the middle point of the bottom side of the boounding box (nx2, n: the number of predicted objects)
        # bx_side_lengths: the length (in pixels) of the sides of the bounding boxes (nx2)
        #                  n: the number of predicted objects,
        #                  first column: the horizontal side,
        #                  second column: the vertical side
        # bx_area: the area of the bounding box in pixels2
        cm_bxs, bx_side_lengths, bx_area = eurekaRes_utils.get_cm(bboxes)

        # keep the bounding boxes whose the area is less than the area threshold (to exclude large objects)
        bboxes = bboxes[bx_area < area_threshold, :]

        # keep the predicted labels whose bounding boxes have an area less than the area threshold
        predicted_labels = predicted_labels[bx_area < area_threshold]

        # keep the objects' coordinates whose bounding boxes have an area less than the area threshold
        cm_bxs = cm_bxs[bx_area < area_threshold, :]

        # keep the boxes' lengths which the area is less than the area threshold
        bx_side_lengths = bx_side_lengths[bx_area < area_threshold, :]

        # if acquired frames are less than the frame_history_length, append the frame to the qeue. Otherwise, continue to computing the average prediction
        if valid_frame_counter < frame_history_length:
            if bboxes.any():
                # in there is at least one object predicted, just append the results of the latest frame
                bboxes_history += [bboxes]
                cm_history += [cm_bxs]
                prediction_history += [predicted_labels]
                score_history += [scores]
                valid_frame_counter += 1
        else:
            if bboxes.any():
                # if there is at least one object predicted, append the results of the latest frame and compute the average prediction
                bboxes_history += [bboxes]
                cm_history += [cm_bxs]
                prediction_history += [predicted_labels]
                score_history += [scores]

                # compute the average prediction among the frames of history batch
                # bboxes: the bounding boxes
                # cm_bxs: the coordinates (in pixels) of the middle point of the bottom side of the boounding box (nx2, n: the number of predicted objects)
                # predicted_labels: the predicted labels of the detected objects
                # scores: the average scores of the predicted labels
                bboxes, cm_bxs, predicted_labels, scores = eurekaRes_utils.average_prediction(bboxes_history, cm_history, prediction_history, score_history, aacm=area_around_cm, clThr=clf_threshold)

                # delete the results of the oldest frame
                del bboxes_history[0]
                del prediction_history[0]
                del cm_history[0]
                del score_history[0]

            # transform the correspoding coordinates of the detected objects from the image frame(pixels) to the frame of the aruco-board
            realWorld_coord, rwc_check = CameraToWorld(cm_bxs, frame)      
            print(realWorld_coord)
            # get the position (x, y in pixels) of the center of mass of each eye-pupil
            leftEye_cx, leftEye_cy = gazeUt.imgProcessingEye(cap_left, 40)
            rightEye_cx, rightEye_cy = gazeUt.imgProcessingEye(cap_right, 40)

            gaze_coord = np.array([], dtype=np.float).reshape(0, 2)

            # if eye-pupils are detected, predict the position of the gaze
            if gazeUt.checkEyeData([leftEye_cx, leftEye_cy, rightEye_cx, rightEye_cy]):
                cx = regr_cx.predict([[leftEye_cx, rightEye_cx, leftEye_cy, rightEye_cy]])
                cy = regr_cy.predict([[leftEye_cx, rightEye_cx, leftEye_cy, rightEye_cy]])

                # display the gaze on the window as a red dot
                cv2.circle(frame, (int(cx), int(cy) ), 20, (0, 0, 255), -5)

                gaze_coord = np.array([cx, cy])

            # define a numpy array to hold the color of each bounding box (for visualization purposes)
            rbg_clr = np.array([], dtype=int).reshape(0, 3)

            # define a counter
            cc = 0
            # define a variable for the object of interest
            oboi = []

            # check every detected object if it is the object of interest
            for rro in bboxes:
                # if gaze coordinates exist
                if gaze_coord.any():
                    # check if the coordinates of the gaze are inside the bounding box
                    if (gaze_coord[0] >= rro[1]) and (gaze_coord[0] <= rro[3]):
                        if (gaze_coord[1] >= rro[0]) and (gaze_coord[1] <= rro[2]):
                            # if the conditions are met, set the red color to the bounding box 
                            # and identify this object as the object of interest
                            rbg_clr = np.vstack((rbg_clr, np.array([0, 0, 255])))
                            oboi = realWorld_coord[cc]
                        else:
                            # otherwise, set the bounding box to be red
                            rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))
                    else:
                        rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))
                else:
                    rbg_clr = np.vstack((rbg_clr, np.array([255, 0, 0])))

                cc += 1

            # draw the bounding boxes in the frame
            frame = eurekaRes_utils.draw_boxes(frame, bboxes, color_pallete=rbg_clr)
            # add the labels and their scores above the bounding box
            frame = eurekaRes_utils.add_classes_names_to_image(frame, bboxes, predicted_labels, scores, text_colors=rbg_clr)
            # if the real-world coordinates were computed, display them below the bounding box
            if rwc_check:
                frame = eurekaRes_utils.display_real_coord(frame, bboxes, realWorld_coord, text_colors=rbg_clr)

            # oboi = [31, 38]
            # np.delete(realWorld_coord, 0, 0)
            print('oboi: ', oboi)

            # if a connection to the socketStream server was established, update the message fields with the resuts and stream the message
            if everything_ok:
                if bboxes.any():
                    # update the objects' locations
                    sockClient.updateMSG("obj_location", realWorld_coord)

                    # update the bounding boxes
                    sockClient.updateMSG("bboxes", bboxes)

                    # if an object of interest was found update the correspoding field with the real values, otherwise add zeros
                    if oboi is None:
                        sockClient.updateMSG("oboi", np.array(oboi))
                    else:
                        sockClient.updateMSG("oboi", np.zeros(2))
                    # send the message to the server
                    sockClient.sendMsg()

        # Display the resulted frame in a window
        cv2.imshow('frame', frame)
        # cv2.imshow('right', cap_right)
        # cv2.imshow('left', cap_left)
        timings = np.vstack((timings, time.time()-start_time))

        # increase the frame counter
        frame_counter += 1.0

        # # write the flipped frame
        out.write(frame)
        start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

# compute the complete duration of the process
duration = time.time() - all_time_start

# close the communication with the server
sockClient.closeCommunication()

# When everything done, release the capture
out.release()
cv2.destroyAllWindows()

# print the statistics
print("fps: ", frame_counter/duration)
print("average process time: ", np.mean(timings))
print("std process time: ", np.std(timings))
