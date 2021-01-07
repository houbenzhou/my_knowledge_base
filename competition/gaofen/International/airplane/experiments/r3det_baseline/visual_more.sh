#!/bin/bash -e
/home/data/hou/soft/conda/envs/detectron2-gpu/bin/python  \
/home/data/hou/workspaces/my_knowledge_base/my_util/data_devkit/images_devkit/visual/visual_object_detection_voc.py \
--img_path /home/hou/data/hou/experiments/gaofen4/result_data/label_visual/Images \
--label_path /home/hou/data/hou/experiments/gaofen4/result_data/r3det_baseline/result_label_file_more \
--out_path /home/hou/data/hou/experiments/gaofen4/result_data/r3det_baseline/visual_more
