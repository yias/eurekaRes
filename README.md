# eurekaRes

An ANN model for fast object detection (targeting real-time performance). Currently under development

The package uses the MS COCO dataset [link](http://cocodataset.org/#home) for training the model

Lisence: MIT
Copyright (c) 2020 Iason Batzianoulis

## Some dependencies:
python 3.7+
tensorflow 2.1
keras 2.3.1
COCO (2017)
scikit-learn 0.22.2+
matplotlib 3.2.1
pandas 1.0.3
numpy 1.16.4

- developed in VSCode

# example for creating a csv file with information for the data (to be used for loading the images and create the batches)
from inside the folder
```bash
$ python eR_coco/eRes_create_coco_datasets.py --ds_name val --oFile validTest
```