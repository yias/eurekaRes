#!/usr/bin/env python3.7


import os
import sys
import ctypes
import pathlib

import cv2

# import random
# import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


# import eurekaRes utilities
if sys.platform == 'linux':
    sys.path.append(str(pathlib.Path().absolute()) + "/../utils")
else:
    sys.path.append(str(pathlib.Path().absolute()) + "\\..\\utils")

import eurekaRes_utils as eResU

import darknet


def matlplotlib_display_img(img, boxes=None):
    """
    The function displays the image an the bounded boxes
    of the detected objects on a figure of matplotlib 
    """
   
    D = img.w * img.h * img.c
    tmp = np.empty((D,), dtype=np.float32)
    ctypes.memmove(tmp.ctypes.data, img.data, D*np.dtype(np.float32).itemsize)

    tmp = np.reshape(tmp, (img.c, img.h, img.w))*255.0
    tmp = np.moveaxis(tmp, 0, -1)

    # display the image with its annotations
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.axis("off")
    ax.imshow(tmp.astype(np.uint8))

    if boxes is not None:

        colors = np.random.rand(len(r),3)
        counter = 0
        # For each instance in the picture, create a rectangle patch and add it to the axes
        for inst in r:
            x, y, off_x, off_y = inst[2]
            left = (x-off_x/2.)*img.w
            right = (x+off_x/2.)*img.w
            top = (y-off_y/2.)*img.h
            bot = (y+off_y/2.)*img.h
            print(left, right, top, bot)
            # x = img.h - x
            rect = patches.Rectangle((x, y), off_x, off_y, linewidth=1, edgecolor=colors[counter], facecolor='none')
            ax.text(x, y, inst[0].decode("utf-8") , fontdict={'color': colors[counter], 'size': 12, 'weight': 'bold'})
            ax.add_patch(rect)
            counter += 1 

    # darknet.free_image(img)

    plt.show()





if __name__ == "__main__":

    # load the network and the labels
    # net = darknet.load_net((yollo_DIR + '/cfg/yolov3-tiny.cfg').encode('utf8'), (yollo_DIR + '/cfg/yolov3-tiny.weights').encode('utf8'), 0)
    net = darknet.load_net((os.environ["DARKNET_DIR"] + '/cfg/yolov3-tiny.cfg').encode('utf8'), (os.environ["DARKNET_DIR"] + '/cfg/yolov3-tiny.weights').encode('utf8'), 0)
    # net = darknet.load_net((os.environ["DARKNET_DIR"] + '/cfg/yolov3.cfg').encode('utf8'), (os.environ["DARKNET_DIR"] + '/cfg/yolov3.weights').encode('utf8'), 0)
    meta = darknet.load_meta((os.environ["DARKNET_DIR"] + '/cfg/coco.data').encode('utf8'))

    dataFolder = str(pathlib.Path().absolute()) + "/../data/"
    vFileName = "gazeRecordings/recording_20200624_1_world_clean.avi"

    outVFileName = "gazeRecordings/20200624_1_world_gaze_objects_yollo_tiny.mp4"
    # create object to capture the frames from an input
    cap = cv2.VideoCapture(dataFolder + vFileName)
    fourCC = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dataFolder + outVFileName, fourCC, 10.0, (1280, 720))

    clf_threshold = 0.2

    # set the resolution of the frame
    # cap.set(3, 1280)
    # cap.set(4, 720)

    Colors = eResU.random_colors(50)

    timings = np.array([], dtype=np.float64).reshape(0, 1)
    start_time = time.time()
    frame_counter = 0.0
    all_time_start = time.time()

    print('is cap opened:')
    print(cap.isOpened())

    while cap.isOpened():
        try:

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                # write the flipped frame
                # out.write(frame)

                # get the object-detection results from darknet
                r = darknet.darknet_detect(net, meta, frame)
                # print(len(r))
                if r:
                    tt = zip(*r)
                    predicted_labels, scores, bboxes = tt
                    predicted_labels = np.asarray(predicted_labels)
                    scores = np.asarray(scores)
                    predicted_labels = predicted_labels[scores > clf_threshold]
                    bboxes = np.asarray(bboxes)
                    bboxes = bboxes[scores > clf_threshold, :]
                    print("shape predicted_labels", predicted_labels.shape)
                    print(predicted_labels)

                    print("shape bboxes", bboxes.shape)
                    print("boxes: ", bboxes)
                    print("shape scores", scores.shape)
                    print(scores)
                    print(frame.shape)
                    bboxes = eResU.process_bboxes(bboxes)
                    frame = eResU.draw_boxes(frame, bboxes, color_pallete=Colors)

                    frame = eResU.add_classes_names_to_image(frame, bboxes, predicted_labels, scores, text_colors=Colors)

                # # Display the resulting frame

                cv2.imshow('frame', frame)
                out.write(frame)
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
    # print(timings)
    print("average process time: ", np.mean(timings))
    print("std process time: ", np.std(timings))
