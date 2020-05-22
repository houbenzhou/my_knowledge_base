#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/detectron2-gpu-clone/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/inference_detectron2.py \
--train_data_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC \
--train_config_path /home/data/hou/workspaces/detectron2/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml \
--input_image_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images \
--model_path /home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/temp/model_final.pth \
--outpath  /home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/labelTxt