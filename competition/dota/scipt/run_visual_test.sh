#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/visual/visual_dota_testdata.py \
--img_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images \
--label_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/labelTxt \
--out_path /home/data/windowdata/temp/visual_dota_test
