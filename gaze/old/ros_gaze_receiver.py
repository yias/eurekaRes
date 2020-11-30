#!/usr/bin/env python
"""
    developer: Iason Batzianoulis
    maintaner: Iason Batzianoulis
    email: iasonbatz@gmail.com
    description: 
    This scripts is an example on how to use the socketStream server for listening to inputs from a client
"""

import argparse
import numpy as np
import sys
from socketStream_py import socketStream
import time

# import ros-related modules
import rospy


def main(args):
    sockHndlr = socketStream.socketStream(svrIP = args.host, svrPort = args.port, socketStreamMode = 1)
    
    sockHndlr.initialize_msgStruct(["clf_threshold", "obj_location", "obj_name", "oboi"])


    

    everything_ok = False
    if sockHndlr.initialize_socketStream() == 0:
        if sockHndlr.runServer() == 0:
            everything_ok = True
    
    # define ros publisher
    target_pub = rospy.Publisher('robot_arm_motion/target', Float32MultiArray, queue_size = 1)

    obstacles_pub = rospy.Publisher('robot_arm_motion/obstacles', Float32MultiArray, queue_size = 1)


    if everything_ok:
        # counter=0
        while(True):
            try:
                if sockHndlr.socketStream_ok():
                    tt=sockHndlr.get_latest()
                    # print(tt)
                    if tt is not None:
                        print(tt)
                        msgData=tt['obj_location']
                        rt=np.array(msgData, dtype=np.float32)
                        # print(rt.shape)
                
            except KeyboardInterrupt:
                break

    sockHndlr.closeCommunication()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCP server for receiving inputs from a client with socketStream')
    parser.add_argument('--host', type=str, help= 'the IP of the server', default='localhost')
    parser.add_argument('--port', type=int, help= 'the port on which the server is listening', default=10352)
    parser.add_argument('--buffersize', type=int, help= 'the size of the buffer for pakets receiving', default=16)
    args=parser.parse_args()
    main(args)