#!/usr/bin/env python3.7



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
