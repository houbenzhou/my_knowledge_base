#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/conversion/dota_to_voc.py \
--input_dota_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800_gsd \
--out_voc_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800_gsd/VOC1 \
--category None \
--tile_size 800 \
--tile_offset 400

