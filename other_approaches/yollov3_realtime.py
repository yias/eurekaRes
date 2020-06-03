#!/usr/bin/env python3.7

import time
import os
import sys
import ctypes

import cv2

# import random
# import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU

# Root directory of the project
yollo_DIR = os.environ["PY_WS"]+"/object_detection/darknet"
sys.path.append(yollo_DIR + "/python")
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


def darknet_detect(trained_net, img_meta, image, thresh=.1, hier_thresh=.1, nms=.45):
    """
    The funtion performs an object detection on the image and returns the objects found
    with their labels and bounding boxes
    """

    # get image original shape
    image_shape = image.shape

    # define and image object (darknet-based)
    im = darknet.make_image(image_shape[1], image_shape[0], image_shape[2])

    # convert the image from a matrx of integers to a matrix of float numbers
    image = image.astype(np.float32)

    # arrange the image to the format that the darket accepts
    image = np.moveaxis(image, -1, 0)

    # reshpape the matrix to an array and scale the data to be between [0,1]
    image = np.reshape(image, [1, np.prod(image_shape)])[0]/255.

    # set the pointer on the array as the data of the object
    im.data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # resize the image
    # im = darknet.letterbox_image(im, 416, 416)

    # perform a detection
    # darknet.predict(trained_net, im.data)
    darknet.predict_image(trained_net, im)

    # define a pointer for transfering the results of the detection
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)

    # get the detected objects and their bounded boxes
    dets = darknet.get_network_boxes(trained_net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)

    # get the pointer of the detection results
    num = pnum[0]
    if nms:
        # if the pointer is not Null (empty), get the names of the labels
        darknet.do_nms_obj(dets, num, img_meta.classes, nms)

    # get the results sorted according to their highest probability
    res = []
    for j in range(num):
        for i in range(img_meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((img_meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])

    # deallocate the memory
    darknet.free_image(im)
    darknet.free_detections(dets, num)

    return res


if __name__ == "__main__":

    # load the network and the labels
    net = darknet.load_net((yollo_DIR + '/cfg/yolov3-tiny.cfg').encode('utf8'), (yollo_DIR + '/trained_models/yolov3-tiny.weights').encode('utf8'), 0)
    meta = darknet.load_meta((yollo_DIR + '/cfg/coco.data').encode('utf8'))

    # create object to capture the frames from an input
    cap = cv2.VideoCapture(0)

    # set the resolution of the frame
    cap.set(3, 1280)
    cap.set(4, 720)

    Colors = eResU.random_colors(50)

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

            # get the object-detection results from darknet
            r = darknet_detect(net, meta, frame)
            print(r)
            print(len(r))

            tt = zip(*r)
            predicted_labels, scores, bboxes = tt
            print("type predicted_labels", type(predicted_labels))
            print(predicted_labels)
            print(bboxes)
            print(type(bboxes))
            # print(tt)
            # predicted_labels, scores, bboxes = r[0]
            # predicted_labels, scores, bboxes = [(x[0], x[1], x[-1]) for x in r]
            # [print(type(x)) for x in r]
            # print(bboxes)
            # print(predicted_labels)
            # print(scores)
            # predicted_labels = [x[0].decode('utf8') for ]
            
            # frame = eurekaRes_utils.draw_boxes(frame, r['rois'], color_pallete=Colors)

            # frame = eurekaRes_utils.add_classes_names_to_image(frame, r['rois'], r['class_ids'], class_names, r['scores'], text_colors=Colors)

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
    # print(timings)
    print("average process time: ", np.mean(timings))
    print("std process time: ", np.std(timings))
