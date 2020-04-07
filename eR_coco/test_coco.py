#!/usr/bin/env python3.7

import os
import sys

sys.path.append(os.environ["PY_WS"]+"/object_detection/coco/PythonAPI")
# Import COCO
from pycocotools.coco import COCO

import numpy as np

import skimage.io as io

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import pylab


dataDir = os.environ["PY_WS"]+"/object_detection/COCO_dataset"
dataType = 'train2017' #'val2017'
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
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
img = coco.loadImgs(imgIds[0])[0]

print("nb of images in the category", len(imgIds))
print(img['file_name'])

# load and display image
I = io.imread('%s/%s/%s'%(dataDir, dataType, img['file_name']))


# load and display instance annotations
fig, ax = plt.subplots(1)
ax.imshow(I)
ax.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
print(anns)

# for an in anns:
#     tt_box = an.get("bbox")
#     rect = patches.Rectangle(tt_box[:2], tt_box[2], tt_box[3], linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#     # print(an)

coco.showAnns(anns)
plt.show()

annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.show()
