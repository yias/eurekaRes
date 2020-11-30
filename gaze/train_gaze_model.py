#!/usr/bin/env python3.7
"""
@author: Iason Batzianoulis

Record data from the gaze-tracker cameras and train a model for gaze prediction

"""


import os
import sys
import pathlib

sys.path.append(str(pathlib.Path().absolute()) + "/gaze_utils")
from recordGaze import gazeTracking
from trainSVR import mySVRfct

# record data
gazeTracking(showPupilsMarker=True, recordFrames=True, recordPupilsPos=True, recordArucoMarker=True, audioCue=False)

# train the model
mySVRfct('gaze_model/data/coord')
