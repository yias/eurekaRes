#!/usr/bin/env python3.7

import os
import sys

sys.path.append(os.environ["PY_WS"]+"/object_detection/coco/PythonAPI")
# Import COCO
from pycocotools.coco import COCO

import numpy as np

import skimage.io as io

import matplotlib.pyplot as plt
import pylab


dataDir = os.environ["PY_WS"]+"/object_detection/COCO_dataset"
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
captionsFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
print("type of annotation file", type(annFile))

# initialize COCO api for instance annotations
coco = COCO(annFile)

# initialize COCO api for captions annotations
coco_caps = COCO(captionsFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' \n'.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['apple', 'wine glass'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

print("nb of images in the category", len(imgIds))

# load and display image
I = io.imread('%s/%s/%s'%(dataDir, dataType, img['file_name']))
plt.axis('off')
# plt.imshow(I)




# load and display instance annotations
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for an in anns:
    print(an)

coco.showAnns(anns)
# plt.show()

annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.show()
