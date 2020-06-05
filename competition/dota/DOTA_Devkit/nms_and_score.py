# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import argparse
import os
import shutil

import numpy as np
import yaml
from dotmap import DotMap


def nms(dets, nms_thresh):
    """
    nms,去除重复框
    """
    # x1、y1、x2、y2、以及score赋值
    # x1 = dets[:, 3]
    # y1 = dets[:, 4]
    # x2 = dets[:, 5]
    # y2 = dets[:, 6]
    # scores = dets[:, 2]
    # x1 = np.array(list(map(float,x1)))
    # y1 = np.array(list(map(float,y1)))
    # x2 = np.array(list(map(float,x2)))
    # y2 = np.array(list(map(float,y2)))
    # scores = np.array(list(map(float,scores)))
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    x1 = np.array(list(map(float, x1)))
    y1 = np.array(list(map(float, y1)))
    x2 = np.array(list(map(float, x2)))
    y2 = np.array(list(map(float, y2)))
    scores = np.array(list(map(float, scores)))

    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## 按照score置信度降序排序
    order = scores.argsort()[::-1]
    keep = []  # 保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留该类剩余box中得分最高的一个
        # 得到相交区域，左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交面积，不重叠的时候为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]  # 因为ovr的数组长度比order数组少一个，所以这里要将所有下标后移一位
    return keep


def nms_score_thresh(lable_file, nms_thresh, score_thresh, categoty_names, out_lable_file):
    # 从label_file中取出每一条记录写入数组
    all_boxes = []
    with open(lable_file, "r", encoding="utf-8", errors="ignore") as file_in:

        while True:
            line = file_in.readline()
            if not line:
                break
            all_boxes.append(line)

    # 将all_boxs每一条记录依据类别进行处理
    with open(out_lable_file, 'a') as file_out:
        for cls_ind, cls in enumerate(categoty_names[0:]):
            all_boxes_temp = []
            for i in all_boxes:
                if (str(cls) == i.split(" ")[0]) & (float(i.split(" ")[2]) >= float(score_thresh)):
                    temp_single_box = []
                    temp_single_box.append(float(i.split(" ")[3]))
                    temp_single_box.append(float(i.split(" ")[4]))
                    temp_single_box.append(float(i.split(" ")[5]))
                    temp_single_box.append(float(i[:-2].split(" ")[6]))
                    temp_single_box.append(float(i.split(" ")[2]))
                    temp_single_box.append(i.split(" ")[1])
                    all_boxes_temp.append(temp_single_box)
            all_boxes_temp = np.array(all_boxes_temp)
            if (all_boxes_temp != np.array([])):
                keep = nms(all_boxes_temp, nms_thresh)
                all_boxes_temp = all_boxes_temp[keep, :]
            for bbox_sore in all_boxes_temp:
                outline = cls + ' ' + bbox_sore[5] + ' ' + str(float(bbox_sore[4])) + ' ' + str(
                    float(bbox_sore[0])) + ' ' + str(float(bbox_sore[1])) + ' ' + str(float(bbox_sore[2])) + ' ' + str(
                    float(bbox_sore[3]))
                file_out.write(outline + '\n')


def get_classname(train_data_path):
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        voc_config = DotMap(config_dict)
        classes = voc_config.dataset.get('classes')
        del (classes[0])
    return classes


def get_parser():
    parser = argparse.ArgumentParser(description="Post-process the prediction results by nms and score threshold")
    parser.add_argument(
        "--lable_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/600_8_16_32_s600/labelTxt",
        help="path to prediction results directory",
    )

    parser.add_argument(
        "--nms_thresh",
        default=0.3,
        help="nms thresh",
    )

    parser.add_argument(
        "--score_thresh",
        default=0.3,
        help="score thresh",
    )
    parser.add_argument(
        "--categoty_names",
        default='/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/newVoc_planelast/VOC',
        help="The category you want to deal with",
    )

    parser.add_argument(
        "--out_lable_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/dota/out/2020_05_14/600_8_16_32_s600/outlabel/labelTxt',
        help="max iter",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    lable_path = args.lable_path
    nms_thresh = args.nms_thresh
    score_thresh = args.score_thresh
    categoty_names = args.categoty_names
    out_lable_path = args.out_lable_path
    categoty_names = get_classname(categoty_names)
    if os.path.exists(out_lable_path):
        shutil.rmtree(out_lable_path)
    if not os.path.exists(out_lable_path):
        os.makedirs(out_lable_path)
    label_names = os.listdir(lable_path)
    for label_name in label_names:
        lable_file = os.path.join(lable_path, label_name)
        out_lable_file = os.path.join(out_lable_path, label_name)
        nms_score_thresh(lable_file, float(nms_thresh), float(score_thresh), categoty_names, out_lable_file)
