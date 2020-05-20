#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/detectron2-gpu-clone/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/inference_detectron2_visual.py \
--input_data /home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images \
--model_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_19/800_4_8_16_32_64_s800/saved_model/saved_model.sdm \
--out_data /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_19/800_4_8_16_32_64_s800/labelTxt
