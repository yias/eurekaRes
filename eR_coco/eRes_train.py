#!/usr/bin/env python3.7
"""
The module of the model or eurekaRes
"""

# import system modules
import os
import sys

# import tensorflow for ANN interface
import tensorflow as tf

# import other helping modules
import json
import time
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# import eurekaRes utilities
sys.path.append(os.environ["PY_WS"]+"/object_detection/eurekaRes/utils")
import eurekaRes_utils as eResU

# import the images from csv file (on proper tf format)

# properly set the ANN to accept the images as tensors and the correspoding labels

# finalize ANN architecture

# train ANN

# test ANN on real-time and measure its performance
