#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/nms_and_score.py \
--lable_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/600_8_16_32_s600/labelTxt \
--nms_thresh 0.3 \
--score_thresh 0.7 \
--categoty_names /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/newVoc_planelast/VOC \
--out_lable_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/600_8_16_32_s600/outlabel/labelTxt
