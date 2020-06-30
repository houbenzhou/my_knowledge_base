#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/my_util/data_devkit/images_devkit/visual/visual_object_detection_voc.py \
--img_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800_gsd/VOC/Images \
--label_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800_gsd/VOC/Annotations \
--out_path /home/data/windowdata/temp/visual_voc
