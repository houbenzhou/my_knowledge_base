#!/bin/bash -e
/home/data/hou/soft/miniconda3/envs/iobjectspy-gpu/bin/python \
/home/data/hou/workspaces/my_knowledge_base/competition/iobjects_tf_faster_rcnn/inference_tf_faster_rcnn.py \
--input_data /home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images \
--model_path /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020-05-08/1024_s800_4_8_16_32/saved_model/saved_model.sdm \
--out_data /home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020-05-08/1024_s800_4_8_16_32/labelTxt
