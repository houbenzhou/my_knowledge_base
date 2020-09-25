#!/usr/bin/env python
# encoding: utf-8
import logging
import os
import sys
import time

import cv2
import numpy as np
import rasterio
import yaml
from dotmap import DotMap
from rasterio import transform
from rasterio.plot import reshape_as_image
from rasterio.windows import Window

from competition.mmdetection_r3det_devkit.tools._inference_r3det import R3detEstimation

"""
影像数据目标检测
"""

if __name__ == '__main__':
    model_path = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200616/latest.pth'
    cfg = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py'

    classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court', 'basketball-court',
               'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor',
               'swimming-pool', 'helicopter']
    input_data = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/dotav1_test/images/P0006.png'
    category_name = classes
    out_data = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs'
    out_name = 'out_file'
    nms_thresh = 0.3
    score_thresh = 0.5
    r3det_inference = R3detEstimation(model_path, cfg, classes)
    r3det_inference.estimation_img(input_data, category_name, out_data, out_name, nms_thresh=0.3,
                                   score_thresh=0.5)
