#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/visual/visual_dota_traindata.py \
--img_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/images \
--label_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/labelTxt \
--out_path /home/data/windowdata/temp/visual_dota_train3
