"""
@author: valentin morel

detection of the Auruco marker with the world camera
in order to take data to train the SVR regression

"""

from __future__ import print_function
from WorldCameraOpening import WorldCameraFrame
import cv2
import cv2.aruco as aruco
import numpy as np


# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

def Worldcoord():
#while(True):
    # Capturing each frame of our video stream
    rgbImage = WorldCameraFrame()
    
    # grayscale image
    grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    #ret, thresholdGrayImage = cv2.threshold(grayImage,100,255,cv2.THRESH_BINARY)
    
    
    
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grayImage, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
    if type(ids) == np.ndarray:
        if ids.size == 1:
            if (ids[0] == 0):
                #print('ID: {}; Corners: {}'.format(ids, corners))
                cx_world = corners[0][0][0][0]
                cy_world = corners[0][0][0][1]
                #print('cx_world: ', cx_world)
                #print('cy_world: ', cy_world)
            
                # Outline all of the markers detected in our image
                rgbImage = aruco.drawDetectedMarkers(rgbImage, corners, ids, borderColor=(0, 0, 255))
                
                flag_data = True
        
                return(cx_world, cy_world, rgbImage, flag_data)
        else:
            
            # Outline all of the markers detected in our image
            rgbImage = aruco.drawDetectedMarkers(rgbImage, corners, ids, borderColor=(0, 0, 255))
            flag_data = False
            
            return(0, 0, rgbImage, flag_data)
            
    else:
        # Outline all of the markers detected in our image
        rgbImage = aruco.drawDetectedMarkers(rgbImage, corners, ids, borderColor=(0, 0, 255))
        flag_data = False
        
        return(0, 0, rgbImage, flag_data)