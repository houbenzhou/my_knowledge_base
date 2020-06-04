#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/dota_task/dota_task2.py \
--input_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/800_2_4_8_16_32_s600/labelTxt \
--out_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/800_2_4_8_16_32_s600/task2 \
--categotys /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/800_2_4_8_16_32_s600/saved_model/saved_model.sdm
