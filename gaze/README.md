# Object dection and gaze tracking

This folder contains the integration of eurekaRes with Mask R-CNN and the gaze-tracker from Pupil-labs for real-time object detection and object-of-interest identification

## installation

- Clone and install Mask R-CNN from this [link](https://github.com/matterport/Mask_RCNN)

- Clone and install the gaze tracking utils from this [link](https://github.com/epfl-lasa/gaze_tracking_object_recognition)

- Clone and install socketStream from this [link](https://github.com/yias/socketStream)

The scripts require python 3.7+. The other requirements could be found in the "requirements.txt" file inside the folder "other". To install the rest of the requirements, run:

```bash
$ python -m pip install -r other/requirements.txt
```

## running the script

- To collect data and train a model for the gaze prediction, run the train_gaze_model.py from the current folder

```bash
$ python train_gaze_model.py
```
Once the script is launched, three windows will pop-up showing captured frames from the world-camera and the two eye-cameras.

Adjust the eye-cameras so that the pupils are approximatelly at the center of the correspondin windows. Press Enter in the terminal to start recording the data, while moving an aruco marker inside the workspace of the world-camera. Once the required number of samples are recorded, the training of the model will start. Once the training is completed, three windows displaying the results will pop-up. Close the windows, to terminate the process.

- To run the real-time detection run:

```bash
$ python gaze_objDect_rt.py
```
If you want to stream the results over the network, a socketStream server should be launched in advance (before launching the gaze_objDect_rt.py script)

## Samurai PC

The Samurai PC of LASA lab, EPFL is already set-up for this project.

First, activate the environment

```bash
$ conda activate iason_env
```

Then, navigate to the folder "gaze" of the eurekaRes:

```bash
$ cd /home/crowdbot/iason_ws/py_ws/eurekaRes/gaze
```