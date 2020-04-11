#!/usr/bin/env python3.7

"""
	Developer: Iason Batzianoulis
	Maintaner: Iason Batzianoulis
	email: iasonbatz@gmail.com

    Description:
    The module contains a function that reads the dimensions and annotations of the images of the categories of interest,
    preprocess the annotation boxes and generate a csv file with the proper format to be used for loading the images when training an ANN.

    MIT License

    Copyright (c) 2020 Iason Batzianoulis

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import os
import sys
import errno
import argparse
from argparse import RawTextHelpFormatter


# import other helping modules
import json
import time
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU

# Import COCO utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/coco/PythonAPI")
from pycocotools.coco import COCO




def create_ds(ds_name, cats_of_interest, outputFileName=None, outputPath=None, show_plot=False):
    """
    Function to create dataset. The function reads the dimensions and annotations of the images of the categories of interest,
    preprocess the annotation boxes and generate a csv file with the proper format to be used for training.

    Arguments
    ---------------------------------------------------------------------------
    ds_name :           the name of the dataset. It could be "train" or "valid"
    cats_of_interest :  a list with name of the categories of interest
    outputFileName :    the name of the output file. The default name is "[ds_name]_dataset.csv"
    outputPath :        the path of the directory to save the output file. The default directory is a folder in the name "datasets" in the current path
    show_plot :         a binary variable indicating if the images will be displayed or not (True: they will be displayed, False: they will not)

    Output
    ---------------------------------------------------------------------------
    The collumns of the csv file will correspond to:
    filename | class_id | class_label | cl_digit[0] | ...  | cl_digit[n] | xmin | ymin | xmax | ymax | scale | padding | window

    correspodence:
    filename ->                             the name of the jpg image
    class_id ->                             the class label in binary format as a sting
    class_label ->                          the class label in decimal format
    cl_digit[0] | ...  | cl_digit[n] ->     the digits of the binary class label, where n is the number of the categories of interest (i.e. classes)
    xmin ->                                 the x coordinate of the annotation box
    ymin ->                                 the y coordinate of the annotation box (xmin and ymin correspond to the bottom left corner of the annotation box)
    xmax ->                                 the x coordinate of the annotation box
    ymax ->                                 the y coordinate of the annotation box (xmin and ymin correspond to the top right corner of the annotation box)
    scale ->                                the scale with which the image should be resized to match the resolution indicated in the config file
    padding ->                              the number of pixels that need to be padded in the original image to match the resolution indicated in the config file
    window ->                               the pixel coordinates of the original image in the final image

    """

    # define the folder that contains the data
    dataType = ds_name + '2017'

    # define root directory
    dataDir = os.environ["PY_WS"]+"/object_detection/COCO_dataset"

    # define the file that contains the COCO annotations
    train_annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    # define the file that contains the COCO labels
    config_file = "coco_config.json"


    # define the output path and file name
    if outputFileName is None:
        output_file_name = ds_name + '_dataset.csv'
        if outputPath is None:
            # create a folder to store the csv files if doesn't exists
            try:
                os.makedirs(os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/" + "datasets")
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            output_save_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/" + "datasets/" + output_file_name

        else:
            output_save_path = outputPath + output_file_name
    else:
        output_file_name = outputFileName + '.csv'
        if outputPath is None:
            output_save_path = os.environ["PY_WS"]+"/object_detection/" + "eurekaRes/" + "datasets/" + output_file_name
        else:
            output_save_path = outputPath + output_file_name

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
    class_digits_label = ["cl_digit " + str(i) for i in range(len(train_catIds))]

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

        for img_nb in train_imgIds:
            # load instance annotations and the image info
            train_annIds = coco.getAnnIds(imgIds=img_nb, catIds=train_catIds, iscrowd=None)
            train_anns = coco.loadAnns(train_annIds)
            img_info = coco.loadImgs(img_nb)

            # create the path of the image
            file_path = dataDir + "/" + dataType + "/"
            file_name = img_info[0].get("file_name")
            # print("image file name: ", file_name)

            image_shape = (img_info[0].get("height"), img_info[0].get("width"))

            # compute the molding specifications for the image
            window, scale, padding = eResU.find_image_molding(image_shape, config_json.get("max_height"), \
                                    config_json.get("max_width"), config_json.get("do_padding"))

            # find all the annotations that are related with the current category (i.e. class)
            # and it's not crowd (in the case of the category "person")
            train_anns_label = [x for x in train_anns if x.get("category_id") == catId and x.get("iscrowd") == 0]

            # create an numpy ndarray with all the boxes of the category (i.e. class)
            img_cat_boxes = np.array([np.array(x.get("bbox")) for x in train_anns_label])

            # mold the annotation boxes
            new_boxes = eResU.mold_ann_boxes(img_cat_boxes, scale, padding, config_json.get("do_padding"))

            # throw each annotation box to df_rows with all the data that correspond to the annotation box
            for inst_box in new_boxes:
                label = np.concatenate((class_label, inst_box)).astype(np.float64)
                str_label = " ".join(format(x, ".5f") for x in label)
                df_rows = df_rows + [[file_name, str_label, str(label_index-1)] + list(map(str, class_label)) +\
                                    list(map(str, inst_box)) + [str(scale), str(padding), str(window)]]

            if show_plot:
                try:
                    # read image from file
                    I = io.imread(file_path+file_name)
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

                    # mold the image
                    start_t = time.time()
                    new_img = eResU.fast_mold_image(I, scale, padding, config_json.get("do_padding"))
                    end_t = time.time()
                    timings = np.vstack([timings, end_t - start_t])

                    # display the molded figure with all the boxes
                    ax2.imshow(new_img)
                    ax2.axis('off')
                    ax2.set_title('molded image with annotations')

                    for inst_box in new_boxes:
                        # df_rows = df_rows + [[file_name, str(class_label)] + list(map(str, inst_box))]
                        rect = patches.Rectangle(inst_box[:2], inst_box[2], inst_box[3], linewidth=1, edgecolor='r', facecolor='none')
                        ax2.add_patch(rect)

                    fig.suptitle("class: " + train_labels[label_index - 1] + "\nImage " + str(img_nb), fontsize=16)
                    plt.show()
                except KeyboardInterrupt:
                    break

    # clear coco variable for saving memory
    coco = []

    # throw the rows to pandas DataFrame object and define the legends of the collumns
    df = pd.DataFrame(df_rows, columns=["filename", "class_label", "class_id"] + class_digits_label +["xmin", "ymin", "xmax", "ymax", "scale", "padding", "window"])

    # same the data to a csv file
    df.to_csv(output_save_path)

    print("output file generated in: ", output_save_path)

    if show_plot:
        print("average comp time = ", np.mean(timings))
        print("std = ", np.std(timings))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Program to create datasets from COCO', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--ds_name', type=str, help='the name of the dataset to be created. \nOptions: \ntrain: for creating the training set, \nval: for creating the validation set', default='train')

    parser.add_argument('--cats', type=str, nargs='+', help='categories of interest', default=['person', 'objects'])

    parser.add_argument('--oFile', type=str, help='the name of the output csv file', default=None)

    parser.add_argument('--oPath', type=str, help='the directory to save the csv file', default=None)

    parser.add_argument('--show_plots', type=bool, help='set True for displaying the images with the annotations (default: False)', default=False)

    args = parser.parse_args()

    # define if train or validation data
    dataset_name = args.ds_name

    # define categories of interest
    categories_of_interest = args.cats

    # set the plot to appear or not
    is_show_plot = args.show_plots

    # set output file name
    of_Name = args.oFile

    # set output file name
    of_path = args.oPath

    # create a dataset csv file
    create_ds(dataset_name, categories_of_interest, outputFileName=of_Name, outputPath=of_path, show_plot=is_show_plot)
