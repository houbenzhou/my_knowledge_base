#!/usr/bin/env python
# encoding: utf-8
import argparse
import os

from competition.mmdetection_r3det_devkit.tools._inference_r3det import R3detEstimation

"""
影像数据目标检测
"""


def get_parser():
    parser = argparse.ArgumentParser(description="train detectron2")
    parser.add_argument(
        "--model_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/latest.pth",
        help="model path",
    )

    parser.add_argument(
        "--cfg",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit_gaofen_plane.py',
        help="path to configs file",
    )

    parser.add_argument(
        "--input_data",
        default='/home/data/competition/gaofen4/International/airplane/data/dota_format/test/images',
        help="path to input image data directory ",
    )
    parser.add_argument(
        "--classes",
        default=['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
                 'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other'],
        help="category name list",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=800,
        help="tile size ",
    )
    parser.add_argument(
        "--tile_offset",
        type=int,
        default=400,
        help="tile offset size ",
    )
    parser.add_argument(
        "--out_data",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/summit3',
        help="path to model path directory  ",
    )
    parser.add_argument(
        "--out_name",
        default='out_file',
        help="path to model path directory ",
    )
    parser.add_argument(
        "--nms_thresh",
        default=0.3,
        help="path to output directory ",
    )
    parser.add_argument(
        "--score_thresh",
        default=0.5,
        help="path to output directory ",
    )
    parser.add_argument(
        "--out_format",
        default="gaofen",
        help="path to output directory ",
    )

    return parser


if __name__ == '__main__':

    # dotav1
    # model_path = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200616/latest.pth'
    # cfg = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py'
    # classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    #            'small-vehicle', 'large-vehicle', 'ship',
    #            'tennis-court', 'basketball-court',
    #            'storage-tank', 'soccer-ball-field',
    #            'roundabout', 'harbor',
    #            'swimming-pool', 'helicopter']
    # input_data = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/dotav1_test/images/P0006.png'
    # category_name = classes
    # out_data = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs'
    # out_name = 'out_file'
    # nms_thresh = 0.3
    # score_thresh = 0.5

    # plane gaofen4
    # model_path = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/latest.pth'
    # cfg = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit_gaofen_plane.py'
    # classes = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
    #            'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other']
    # input_data = '/home/data/competition/gaofen4/International/airplane/data/dota_format/test/images'
    # category_name = classes
    # out_data = '/home/data/hou/workspaces/my_knowledge_base/competition/mmdetection_r3det_devkit/tools/work_dirs/r3det_r50_fpn_2x_20200923_gaofen_plane/summit2'
    # out_name = 'out_file'
    # nms_thresh = 0.3
    # score_thresh = 0.5
    args = get_parser().parse_args()
    model_path = args.model_path
    cfg = args.cfg
    input_data = args.input_data
    classes = args.classes
    tile_size = args.tile_size
    tile_offset = args.tile_offset
    out_data = args.out_data
    out_name = args.out_name
    nms_thresh = args.nms_thresh
    score_thresh = args.score_thresh
    out_format=args.out_format
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    r3det_inference = R3detEstimation(model_path, cfg, classes, tile_size=800, tile_offset=700)
    # r3det_inference.estimation_img(input_data, out_data, out_name, nms_thresh=0.3,
    #                                score_thresh=0.5)
    r3det_inference.estimation_dir(input_data, out_data,out_format, nms_thresh=0.3,
                                   score_thresh=0.5)
