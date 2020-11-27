"""
@author: Valentin Morel
         Iason Batzianoulis (maintainer)

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

def PolygonSort(corners):
    n = len(corners)
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    cornersWithAngles.sort(key = lambda tup: tup[2])
    return cornersWithAngles
    # return map(lambda (x, y, an): (x, y), cornersWithAngles)


def PolyArea(x,y):
    corners = np.vstack((x,y))
    corners = list(PolygonSort(corners.T))
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
    # return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def marker_area(marker):
    tt = PolyArea(marker[:,0], marker[:,1])
    return tt


def get_cm_area(corners):
    amarkers_cm = np.array([], dtype=float).reshape(0, 2)
    amarkers_area = np.array([], dtype=float)
    print('cc: ', corners[0][0])
    for i in range(len(corners)):
        amarkers_cm = np.vstack((amarkers_cm, np.mean(corners[i][0], axis=0)))
        amarkers_area = np.hstack((amarkers_area, marker_area(corners[i][0])))
    print('aa: ', amarkers_area[0])
    s_order = np.argsort(amarkers_area)
    return amarkers_cm[s_order,:], amarkers_area[s_order]
    

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
    # print('corners: ', corners[0][0].shape)
    # print('ids: ', len(ids))
    
    # print('cms: ', c_cms.shape)
    # print('area: ', m_area)
    if ids is not None and len(ids) >= myLenId:

        # c_cms, m_area = get_cm_area(corners)
        # print('length: ', m_area.shape)
        # print('m_area', m_area)

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