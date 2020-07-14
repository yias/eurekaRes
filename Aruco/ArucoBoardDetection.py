"""
@author: valentin morel

Detection of the auruco board in order to find for each frame the extrinsic
coefficient to determine the pose of the world camera.
--> Rotational matrix and translation vector

"""

from __future__ import print_function
# from WorldCameraOpening import WorldCameraFrame
import cv2
import cv2.aruco as aruco
import numpy as np
from numpy.linalg import inv
import os


# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

#Create grid board object we're using in our stream
#Dimension in cm
board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=2.6, # 3.15
        markerSeparation=0.3, # 0.6
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None
myLenId = 3

current_path = os.path.dirname(os.path.abspath(__file__))
camera_mtx = np.load(current_path + '/mtx_flat.npy') 
dist_coefs = np.load(current_path + '/dist_flat.npy')

def CameraToWorld(in_camera_coord, rgbImage):

    global camera_mtx
    global dist_coefs
    global board

#Uncomment the following if you want to use this script alone
    #while(True):
    # Capturing each frame of our video stream
   # rgbImage = WorldCameraFrame()

    # grayscale image
    grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)

    # print(type(ARUCO_PARAMETERS))
    # Detect Aruco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(grayImage, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    # print(type(ids))
    # print(ids)
    # print(corners)
    # Make sure at least 3 markers are detected
    if ids is not None and len(ids) >= myLenId:


        retval, rvecs, tvecs = aruco.estimatePoseBoard(corners, ids, board, camera_mtx, dist_coefs)
        #print('rvecs: ', rvecs)
        #print('tvecs: ', tvecs)
        #R = RxRyRz

        # Rotation matix
        rmat = cv2.Rodrigues(rvecs)
        R = rmat[0] 
        R1 = R[:, 0]
        R2 = R[:, 1]
        rgbImage = aruco.drawAxis(rgbImage, camera_mtx, dist_coefs, rvecs, tvecs, 3.15)
        # Translation vector
        t = tvecs
        M = np.column_stack((R1, R2, t))

        # C = np.array([[px[0], py[0], 1]])
        # C = np.transpose(C)
        # print(C)
        # print("in_camera_coord: ", in_camera_coord)
        # print(in_camera_coord.shape)
        C = np.hstack([in_camera_coord, np.ones((in_camera_coord.shape[0], 1))])
        C = np.transpose(C)
        # Homography matrix
        AR = np.matmul(camera_mtx, M)
        AR_inv = inv(AR)

        # coordinate in world space multiply by the scalling factor
        P = np.matmul(AR_inv, C)


        # scaling factor
        k = P[2]
        # print("k: ", k)
        x_world = P[0]/ k
        y_world = P[1]/ k
        real_world_coord = P[0:2]
        # print("real_world_coord: ", real_world_coord)
        real_world_coord[0] = real_world_coord[0] / k
        real_world_coord[1] = real_world_coord[1] / k

        # print("x_world type: ", type(x_world))
        # print("x_world shape: ", x_world.shape)
        # print("y_world type: ", type(y_world))
        # print("y_world type: ", y_world.shape)
        real_world_coord = np.flip(real_world_coord.transpose(), 1)
        check = True
        return real_world_coord, check

    else:

        check = False
        return None, check


## Uncomment the following if you want to use this script alone        
#        #print(type(rmat[2]))
#        rgbImage = aruco.drawAxis(rgbImage, camera_mtx, dist_coefs, rvecs, tvecs, 0.0315)
#    # Outline all of the markers detected in our image
#    rgbImage = aruco.drawDetectedMarkers(rgbImage, corners, ids, borderColor=(0, 0, 255))
#    
#    cv2.circle(rgbImage, (px, py), 5, (0, 0, 255), -5)
#    
#    
#    # Display our image
#    cv2.imshow('QueryImage', rgbImage)
    #cv2.imshow('threshold', thresholdGrayImage)
    

# Exit at the end of the video on the 'q' keypress
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cv2.destroyAllWindows()


if __name__ == '__main__':
    CameraToWorld(0, 0, 0)