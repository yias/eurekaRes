"""
@author: valentin morel

Convolution with the eye frames in order to determine the ROI of the pupil

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def checkCoord(coord):
    if coord < 0:
        coord_check = 0
    elif coord > 399:
        coord_check = 400
    elif 0 <= coord < 400:
        coord_check = coord

    return(coord_check)

def convolution(mycap):

    rgbImage = mycap
    grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    
    a=1.1
    
    kernel = np.array(([0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, a, a, 0, 0],
                       [0, 0, a, a, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]), np.float32)

    # convolution of the input gray image with the kernel to detedt pupil ROI    
    output = cv2.filter2D(grayImage, -1, kernel)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(output)
    
    x_min = minLoc[0]
    y_min = minLoc[1]
    myRange = 100
    
    # Define the pupil ROI
    roi_x_min = x_min-myRange
    roi_x_max = x_min+myRange
    roi_y_min = y_min-myRange
    roi_y_max = y_min+myRange
    
    # check if the coordinates are not outside the size of the image (400x400)
    roi_x_min = checkCoord(roi_x_min)
    roi_x_max = checkCoord(roi_x_max)
    roi_y_min = checkCoord(roi_y_min)
    roi_y_max = checkCoord(roi_y_max)
   
    return(roi_x_min, roi_x_max, roi_y_min, roi_y_max)   
    
if __name__ == '__main__':
    convolution()