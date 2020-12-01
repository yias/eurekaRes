"""
@author: Iason Batzianoulis

Open all the cameras of the Eye Tracker display the images and save them to avi files

"""


# from __future__ import print_function

import os
import sys
import select

import logging
logging.basicConfig(level=logging.INFO)
import cv2

import numpy as np
import pathlib

sys.path.append(str(pathlib.Path().absolute()) + "/../Aruco")
print(pathlib.Path().absolute())
print(sys.path)

from EyeCameraOpening import EyeCameraFrame
from ArucoMarkerDetection import Worldcoord
from kernelConvolution import convolution
import gazeTracking_utils as gazeUt

import pandas as pd
import time

def heardEnter():
    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s==sys.stdin:
            input = sys.stdin.readline()
            return True
    return False


def checkData(arr):
    x = np.array(arr)
    if np.where(x==0)[0].size == 0:
        return True
    else:
        return False

def speak(sentence):
    os.system("espeak-ng -v mb-us1 -s 140 '" + sentence +"'")


def gazeTracking(showPupilsMarker=False, recordFrames=False, recordPupilsPos=False, recordArucoMarker=False, folderName=None, datafile='coord', samplesToRecord=800, audioCue=False):
    # Id to determine if a lot of frames are skipped during the tracking of the pupil
    Id = 0  
    
    r_data = []

    header = []

    if recordFrames:

        # get the current dir
        current_dir = os.getcwd()
        
        # if folder name is none set the folder name to data for saving the recorded data
        if folderName is None:
            folderName = current_dir + '/gaze_model/data/'
        else:
            folderName = current_dir + '/' + folderName + '/'

        # if the folder name doesn't exists, create it
        if not os.path.isdir(folderName):
            os.mkdir(folderName)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_left= cv2.VideoWriter(folderName + datafile + '_eye_left.avi', fourcc, 30, (400,400))    
        out_right= cv2.VideoWriter(folderName + datafile + '_eye_right.avi',fourcc, 30, (400,400))
        out_world= cv2.VideoWriter(folderName + datafile + '_world.avi',fourcc, 30, (1280,720))


    if recordPupilsPos or recordArucoMarker:
        header += ['time']

    if recordPupilsPos:
        header += ['left_cx', 'left_cy', 'right_cx', 'right_cy']

        if not recordFrames:
            # get the current dir
            current_dir = os.getcwd()
            
            # if folder name is none set the folder name to data for saving the recorded data
            if folderName is None:
                folderName = current_dir + '/gaze_model/data/'
            else:
                folderName = current_dir + '/' + folderName + '/'

            # if the folder name doesn't exists, create it
            if not os.path.isdir(folderName):
                os.mkdir(folderName)
    
    if recordArucoMarker:
        header += ['aruco_cx', 'aruco_cy']

    startTime = time.time()
    
    isRecordingOn = False

    validSampleCounter = 0

    if recordFrames or recordPupilsPos or recordArucoMarker:
        print('Adjust the eye-cameras to contain the eyes and press Enter to start recording')

    while True:

        try:

            tmp_d = []

            #cap_world = WorldCameraFrame()
            cap_left, cap_right = EyeCameraFrame()

            t_cap_left = cap_left

            t_cap_right = cap_right
            
            # #Record World data from the ArucoMarkerDetection script to detect the ArucoMarker
            cx_world, cy_world, rgbImageWorld, flag_data = Worldcoord()

            if heardEnter():
                isRecordingOn = True
                print('Start recording ...')
                if audioCue:
                    speak('Start recording')

            if isRecordingOn:
                if recordPupilsPos or recordArucoMarker:
                    tmp_d += [time.time() - startTime]

                if showPupilsMarker:
                    leftEye_cx, leftEye_cy = gazeUt.imgProcessingEye(cap_left, 40)
                    # print(leftEye_cx, leftEye_cy)

                    rightEye_cx, rightEye_cy = gazeUt.imgProcessingEye(cap_right, 40)

                    if recordPupilsPos:
                        tmp_d += [leftEye_cx, leftEye_cy, rightEye_cx, rightEye_cy]

                if recordArucoMarker:
                    tmp_d += [cx_world, cy_world]
                    print(tmp_d)
                
                if tmp_d:
                    if checkData(tmp_d):
                        validSampleCounter += 1
                    r_data += [tmp_d]
            
                        
                Id = Id+1                      

            if validSampleCounter > samplesToRecord:
                if isRecordingOn:
                    isRecordingOn = False
                    print('Recording complete')
                    break
            cv2.imshow("Right eye", cap_right)
            cv2.imshow("Left eye", cap_left)

            cv2.imshow('World', rgbImageWorld)

            if isRecordingOn:
                if recordFrames:
                    # record the frames
                    out_left.write(cap_left)
                    out_right.write(cap_right)
                    out_world.write(rgbImageWorld)
        
            cv2.waitKey(1)
        
        except KeyboardInterrupt:
            print("Stopped capturing")
            break
    
    if r_data:
        print("Saving files")
        drame = pd.DataFrame(r_data, columns=header)
        drame.to_csv(folderName + datafile + '.csv')
    
    if recordFrames:
        out_left.release()
        out_right.release()
        out_world.release()

            

if __name__ == '__main__':
    gazeTracking(showPupilsMarker=True, recordFrames=True, recordPupilsPos=True, recordArucoMarker=True, audioCue=False)
    # gazeTracking()

