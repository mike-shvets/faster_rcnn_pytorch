import numpy as np

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

imdb_name = 'spacenet_train'
cfg_file = '../experiments/cfgs/faster_rcnn_end2end.yml'

_DEBUG = True

# load config
cfg_from_file(cfg_file)

# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

blobs = data_layer.forward(random_rotation=True)
im_data = blobs['data']
im_info = blobs['im_info']
gt_boxes = blobs['gt_boxes']
gt_ishard = blobs['gt_ishard']
dontcare_areas = blobs['dontcare_areas']


import cv2
from faster_rcnn.utils.blob import inv_prep_visual

print cv2.imwrite('fig/image0.png', inv_prep_visual(im_data[0]))
np.save('fig/gt0.npy', gt_boxes)

from faster_rcnn.datasets.SpaceNet_utils.gists.plot_truth_coords import plot_truth_coords
from matplotlib import pyplot as plt

img = inv_prep_visual(im_data[0])
boxes = gt_boxes
coords = [np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]]) for x1, y1, x2, y2, _ in boxes]

_ = plot_truth_coords(img, coords, plot_name='fig/res0.png')



