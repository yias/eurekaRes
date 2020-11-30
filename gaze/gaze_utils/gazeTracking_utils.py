"""
@author: Iason Batzianoulis

preprocessing utilities

"""

import cv2
import numpy as np
from kernelConvolution import convolution

def imgProcessingEye(myCap, threshold):
       
    rgbImage = myCap
    
    # define the pupil ROI
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = convolution(rgbImage)
    rgbImageROI= rgbImage[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    
    # Our operations on the frame come here
    grayImage = cv2.cvtColor(rgbImageROI, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
    openImage = cv2.morphologyEx(grayImage, cv2.MORPH_OPEN, kernel)
    
    # define a frame to apply a threshold
    ret, thresholdGrayImage = cv2.threshold(openImage, threshold, 255, cv2.THRESH_BINARY_INV) #
    
    medianBlurredImage = cv2.medianBlur(thresholdGrayImage, 7)
    blurredGaussImage = cv2.GaussianBlur(thresholdGrayImage, (5, 5), 0)
    
    mykernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    openingImage = cv2.morphologyEx(blurredGaussImage, cv2.MORPH_OPEN, mykernel)
    
    # initiate variable for different checks
    nbrEllipse = 0
    pi_4 = np.pi * 4
    old_circularity = -50
    old_area = -50
    
    # find contours in the binary image after blurring and opening
    contours, hierarchy = cv2.findContours(openingImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #im2 table of pixel value

    pupilContour = []
        
    for c in contours:
        area = cv2.contourArea(c)
        arclen = cv2.arcLength(c, True)
        
        if arclen > 0:
            new_circularity = (pi_4 * area) / (arclen * arclen)
        else:
            new_circularity = -100

        # check the circularity of the contour
        if ((new_circularity > old_circularity)):
            pupilContour = c
            old_circularity = new_circularity
            old_area = area
      
    # check if the selected contour can be the pupil
    if len(pupilContour) > 4:
        if ((old_circularity > 0.55) and (6000 > old_area > 200)):
            myEllipse = cv2.fitEllipse(pupilContour)
            nbrEllipse += 1    
           
    if nbrEllipse == 1:
        coordEllipse = myEllipse[0]
        x = coordEllipse[0]
        y = coordEllipse[1]
        cx = int(x) +roi_x_min
        cy = int(y) + roi_y_min

        # fit ellipse on rgb Image                    
        cv2.ellipse(rgbImageROI, myEllipse, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(rgbImage, (cx, cy), 5, (0, 255, 255), -5)
    else:
        cx = 0
        cy = 0

    return cx, cy

def checkEyeData(arr):
    x = np.array(arr)
    if np.where(x==0)[0].size == 0:
        return True
    else:
        return False