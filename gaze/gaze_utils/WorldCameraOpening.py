"""
@author: valentin morel

Open the world camera with the uvc library

"""

import uvc
import logging
logging.basicConfig(level=logging.INFO)
import cv2
import numpy as np
import os

# dev_list =  uvc.device_list()    
# cap_world = uvc.Capture(dev_list[2]['uid'])


dev_list =  uvc.device_list()    
camera_name = "Pupil Cam1 ID2"
dev = [d for d in dev_list if d['name'] == camera_name]
cap_world = uvc.Capture(dev[0]['uid'])

current_path = os.path.dirname(os.path.abspath(__file__))
  
# resolution,FPS
cap_world.frame_mode = (1280,720,60)

# load the intrinsic matrix and  distortion coefficients to undistort the image
camera_mtx = np.load(current_path + '/mtx.npy') 
dist_coefs = np.load(current_path + '/dist.npy') 



def WorldCameraFrame():
   
    frame = cap_world.get_frame_robust()   
    rgbImage = frame.bgr
    
    undst = cv2.undistort(rgbImage, camera_mtx, dist_coefs)
    
    return(undst)

def maintest():
    
    # Uncomment the following to know which mode one can chose for the camera
    #print(cap_world.avaible_modes) 
    
    while True:
      
        undst = WorldCameraFrame()        
        cv2.imshow("undst", undst)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    maintest()