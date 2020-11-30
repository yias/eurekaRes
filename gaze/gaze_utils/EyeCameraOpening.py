"""
@author: valentin morel

Open the two eye cameras with the uvc library

"""

import uvc
import cv2
import logging
logging.basicConfig(level=logging.INFO)

dev_list =  uvc.device_list()
print(dev_list)

camera_name_left = "Pupil Cam2 ID1"
camera_name_right = "Pupil Cam2 ID0"
dev_left = [d for d in dev_list if d['name'] == camera_name_left]
dev_right = [d for d in dev_list if d['name'] == camera_name_right]
cap_left = uvc.Capture(dev_left[0]['uid'])
cap_right = uvc.Capture(dev_right[0]['uid'])

# cap_left = uvc.Capture(dev_list[0]['uid'])
# cap_right = uvc.Capture(dev_list[1]['uid'])

# configure the Pupil 200Hz IR cameras:
controls_dict_left = dict([(c.display_name, c) for c in cap_left.controls])
controls_dict_right = dict([(c.display_name, c) for c in cap_right.controls])
controls_dict_left['Auto Exposure Mode'].value = 1
controls_dict_right['Auto Exposure Mode'].value = 1
controls_dict_left['Gamma'].value = 200
controls_dict_right['Gamma'].value = 200

# resolution,FPS
cap_left.frame_mode = (400,400,60)
cap_right.frame_mode = (400,400,60)

def EyeCameraFrame():
   
    frame_left = cap_left.get_frame_robust()   
    rgbImageLeft = frame_left.bgr
    
    frame_right = cap_right.get_frame_robust()   
    rgbImageRight = frame_right.bgr
        
    return(rgbImageLeft, rgbImageRight)

def maintest():
    
    # Uncomment the following to know which mode one can chose for the camera
    #print(cap_right.avaible_modes) 
    
    while True:
    
        rgbImageLeft, rgbImageRight = EyeCameraFrame()        
        cv2.imshow("rgbImageLeft", rgbImageLeft)
        cv2.imshow("rgbImageRight", rgbImageRight)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    maintest()