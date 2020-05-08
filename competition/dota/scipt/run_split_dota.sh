#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/split_dota_data.py \
--input_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val \
--out_path /home/data/windowdata/data/dota/dotav1/dotav1/train_val/test \
--gap 100 \
--subsize 800 \
--thresh 0.7 \
--rate 1

