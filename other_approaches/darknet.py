#!/usr/bin/env python3.7

import os
import sys
import ctypes
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("w", ctypes.c_float),
                ("h", ctypes.c_float)]

class DETECTION(ctypes.Structure):
    _fields_ = [("bbox", BOX),
                ("classes", ctypes.c_int),
                ("prob", ctypes.POINTER(ctypes.c_float)),
                ("mask", ctypes.POINTER(ctypes.c_float)),
                ("objectness", ctypes.c_float),
                ("sort_class", ctypes.c_int)]


class IMAGE(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int),
                ("h", ctypes.c_int),
                ("c", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_float))]

class METADATA(ctypes.Structure):
    _fields_ = [("classes", ctypes.c_int),
                ("names", ctypes.POINTER(ctypes.c_char_p))]

    
if sys.platform == 'linux':
    lib = ctypes.CDLL(os.environ["DARKNET_DIR"] + "/libdarknet.so", ctypes.RTLD_GLOBAL)
else:
    lib = ctypes.CDLL(os.environ["DARKNET_DIR"] + "\\libdarknet.so", ctypes.RTLD_GLOBAL)

lib.network_width.argtypes = [ctypes.c_void_p]
lib.network_width.restype = ctypes.c_int
lib.network_height.argtypes = [ctypes.c_void_p]
lib.network_height.restype = ctypes.c_int

predict = lib.network_predict
predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
predict.restype = ctypes.POINTER(ctypes.c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [ctypes.c_int]

make_image = lib.make_image
make_image.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
get_network_boxes.restype = ctypes.POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [ctypes.c_void_p]
make_network_boxes.restype = ctypes.POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

network_predict = lib.network_predict
network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [ctypes.c_void_p]

load_net = lib.load_network
load_net.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
load_net.restype = ctypes.c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, ctypes.c_int, ctypes.c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [ctypes.c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [ctypes.c_void_p, IMAGE]
predict_image.restype = ctypes.POINTER(ctypes.c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    
    im = load_image(image, 0, 0)
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)
    print('t1')
    predict_image(net, im)
    print('t2')
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    print('t3')
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def darknet_detect(trained_net, img_meta, image, thresh=.1, hier_thresh=.1, nms=.45):
    """
    The funtion performs an object detection on the image and returns the objects found
    with their labels and bounding boxes
    """

    # get image original shape
    image_shape = image.shape

    # define and image object (darknet-based)
    im = make_image(image_shape[1], image_shape[0], image_shape[2])

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
    predict_image(trained_net, im)

    # define a pointer for transfering the results of the detection
    num = ctypes.c_int(0)
    pnum = ctypes.pointer(num)

    # get the detected objects and their bounded boxes
    dets = get_network_boxes(trained_net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)

    # get the pointer of the detection results
    num = pnum[0]
    if nms:
        # if the pointer is not Null (empty), get the names of the labels
        do_nms_obj(dets, num, img_meta.classes, nms)

    # get the results sorted according to their highest probability
    res = []
    for j in range(num):
        for i in range(img_meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((img_meta.names[i].decode('utf8'), dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    
    # deallocate the memory
    # free_image(im)
    
    free_detections(dets, num)

    return res


if __name__ == "__main__":

    darknet_img_shape = (416, 416)
    net = load_net((os.environ["DARKNET_DIR"] + '/cfg/yolov3-tiny.cfg').encode('utf8'), (os.environ["DARKNET_DIR"] +  '/cfg/yolov3-tiny.weights').encode('utf8'), 0)
    # net = load_net((os.environ["DARKNET_DIR"] + '/cfg/yolov3.cfg').encode('utf8'), (os.environ["DARKNET_DIR"] +  '/cfg/yolov3.weights').encode('utf8'), 0)
    meta = load_meta((os.environ["DARKNET_DIR"] + '/cfg/coco.data').encode('utf8'))
    r = detect(net, meta, (os.environ["DARKNET_DIR"] + '/data/dog.jpg').encode('utf8'))
    print(r)


    img = load_image((os.environ["DARKNET_DIR"] + '/data/dog.jpg').encode('utf8'), 0, 0)
    # img = letterbox_image(img, darknet_img_shape[0], darknet_img_shape[1])
    print(img.w, img.h, img.c)


    # scale_factor_x = (img.w - 1)/(darknet_img_shape[0] - 1)
    # scale_factor_y = (img.h - 1)/(darknet_img_shape[1] - 1)

    D = img.w * img.h * img.c
    tmp = np.empty((D,), dtype=np.float32)
    ctypes.memmove(tmp.ctypes.data, img.data, D*np.dtype(np.float32).itemsize)

    tmp = np.reshape(tmp, (img.c, img.h, img.w))*255.0
    tmp = np.moveaxis(tmp, 0, -1)

    # display the image with its annotations
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.axis("off")
    ax.imshow(tmp.astype(np.uint8))

    colors = np.random.rand(len(r), 3)
    counter = 0
    # For each instance in the picture, create a rectangle patch and add it to the axes
    for inst in r:
        x, y, off_x, off_y = inst[2]
        ll, bb, rr, tt = inst[2]
        # print(img.w, img.h)
        # left  = (x-off_x/2.)*img.w
        # right = (x+off_x/2.)*img.w
        # top   = (y-off_y/2.)*img.h
        # bot   = (y+off_y/2.)*img.h
        left = x * 416 / img.w
        right = off_x * 416 / img.w
        top = off_y * 416 / img.h
        bot = y * 416 / img.h
        # left = img.h - (x * img.w / 416)
        # right = img.h - (off_x * img.w / 416)
        # top = img.w - (y * img.h / 416)
        # bot = img.w - (off_y * img.h / 416)

        print(left, right, top, bot)
        # x = img.h - x
        # rect = patches.Rectangle((x, y), off_x, off_y, linewidth=1, edgecolor=colors[counter], facecolor='none')
        # ax.text(x, y, inst[0].decode("utf-8"), fontdict={'color': colors[counter], 'size': 12, 'weight': 'bold'})
        rect = patches.Rectangle((x - (off_x/2), y - (off_y/2)), off_x, off_y, linewidth=1, edgecolor=colors[counter], facecolor='none')
        ax.text(x - (off_x/2), y - (off_y/2), inst[0].decode("utf-8"), fontdict={'color': colors[counter], 'size': 12, 'weight': 'bold'})
        ax.add_patch(rect)
        counter += 1 

    free_image(img)

    plt.show()
