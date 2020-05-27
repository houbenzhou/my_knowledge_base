#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/detectron2-gpu-clone/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/train_detectron2.py \
--train_data_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC \
--train_config_path /home/data/hou/workspaces/detectron2/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml \
--weight_path /home/data/hou/workspaces/detectron2/data/model/model/ablations_for_deformable_conv_and_cascade_rcnn/cascade_mask_rcnn_R_50_FPN_3x/model_final_480dd8.pkl \
--max_iter 16000 \
--out_dir /home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/temp
