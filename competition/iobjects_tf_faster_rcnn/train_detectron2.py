# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
# dota
import argparse

from competition.iobjects_tf_faster_rcnn._detectron2_model.iobjectspy_tools import get_classname, \
    register_all_pascal_voc
from competition.iobjects_tf_faster_rcnn._detectron2_model.train_detectron2 import train_iobjectspy_voc


def get_parser():
    parser = argparse.ArgumentParser(description="train detectron2")
    parser.add_argument(
        "--train_data_path",
        default="/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC",
        help="path to train data directory",
    )

    parser.add_argument(
        "--train_config_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml',
        help="path to config file",
    )

    parser.add_argument(
        "--weight_path",
        default='/home/data/hou/workspaces/detectron2/data/model/model/ablations_for_deformable_conv_and_cascade_rcnn/cascade_mask_rcnn_R_50_FPN_3x/model_final_480dd8.pkl',
        help="path to pre training model ",
    )

    parser.add_argument(
        "--out_dir",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/2020_05_26/dota/model4',
        help="path to output directory",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="max iter",
    )

    parser.add_argument(
        "--ml_set_tracking_path",
        default="file:///home/data/windowdata/mlruns",
        help="set tracking path",
    )

    parser.add_argument(
        "--experiment_id",
        default="detectron2_dota",
        help="experiment",
    )

    parser.add_argument(
        "--ml_experiment_tag",
        default="dota_splite800_2020_07_01",
        help="experiment tag",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    train_data_path = args.train_data_path
    train_config_path = args.train_config_path
    weight_path = args.weight_path
    max_iter = args.max_iter
    out_dir = args.out_dir
    ml_set_tracking_path = args.ml_set_tracking_path
    experiment_id = args.experiment_id
    ml_experiment_tag = args.ml_experiment_tag

    data_path_name = train_data_path.split("/")[-1]
    class_names = get_classname(train_data_path)
    register_all_pascal_voc(train_data_path=train_data_path, class_names=class_names)
    register_train_name = data_path_name + '_trainval'

    train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name,
                         ml_set_tracking_path, experiment_id, ml_experiment_tag)