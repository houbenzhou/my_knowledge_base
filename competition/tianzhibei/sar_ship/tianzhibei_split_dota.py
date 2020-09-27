import argparse
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio
from PIL import Image
from PIL import ImageFile
from rasterio.windows import Window


def split_dota2traindata(input_img_path, input_label_path,
                         out_train_data_img, out_train_data_label, out_val_data_img, out_val_data_label,
                         out_test_data_img, out_test_data_label):
    label_names_list = os.listdir(input_label_path)
    train_length = int((len(label_names_list) / 5) * 4)
    val_length = int(len(label_names_list) / 10)
    list_train = label_names_list[0:train_length]
    list_val = label_names_list[train_length:train_length + val_length]
    list_test = label_names_list[train_length + val_length:]
    # train_split
    for label_name in list_train:
        full_label_name = os.path.join(input_label_path, label_name)
        if os.path.isfile(full_label_name):
            shutil.copy(full_label_name, out_train_data_label)
        img_name = label_name.split('.')[0] + ".png"
        full_img_name = os.path.join(input_img_path, img_name)
        if os.path.isfile(full_img_name):
            shutil.copy(full_img_name, out_train_data_img)
    # val_split
    for label_name in list_val:
        full_label_name = os.path.join(input_label_path, label_name)
        if os.path.isfile(full_label_name):
            shutil.copy(full_label_name, out_val_data_label)
        img_name = label_name.split('.')[0] + ".png"
        full_img_name = os.path.join(input_img_path, img_name)
        if os.path.isfile(full_img_name):
            shutil.copy(full_img_name, out_val_data_img)
    # test_split
    for label_name in list_test:
        full_label_name = os.path.join(input_label_path, label_name)
        if os.path.isfile(full_label_name):
            shutil.copy(full_label_name, out_test_data_label)
        img_name = label_name.split('.')[0] + ".png"
        full_img_name = os.path.join(input_img_path, img_name)
        if os.path.isfile(full_img_name):
            shutil.copy(full_img_name, out_test_data_img)


def get_parser():
    parser = argparse.ArgumentParser(description="tianzhibei_sar_to_dota")

    parser.add_argument(
        "--input_img_path",
        default="/home/hou/Desktop/windowdata/temp2/images",
        help="input images data path",
    )

    parser.add_argument(
        "--input_label_path",
        default="/home/hou/Desktop/windowdata/temp2/labelTxt",
        help="input labels data path",
    )

    parser.add_argument(
        "--out_path",
        default='/home/hou/Desktop/windowdata/4968465',
        help="Output base path for dota data",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_img_path = args.input_img_path
    input_label_path = args.input_label_path
    out_path = args.out_path
    out_train_data = os.path.join(out_path, "train")
    out_val_data = os.path.join(out_path, "val")
    out_test_data = os.path.join(out_path, "test")
    out_train_data_img = os.path.join(out_train_data, "images")
    out_train_data_label = os.path.join(out_train_data, "labelTxt")
    out_val_data_img = os.path.join(out_val_data, "images")
    out_val_data_label = os.path.join(out_val_data, "labelTxt")
    out_test_data_img = os.path.join(out_test_data, "images")
    out_test_data_label = os.path.join(out_test_data, "labelTxt")
    if not os.path.exists(out_train_data_img):
        os.makedirs(out_train_data_img)
    if not os.path.exists(out_train_data_label):
        os.makedirs(out_train_data_label)
    if not os.path.exists(out_val_data_img):
        os.makedirs(out_val_data_img)
    if not os.path.exists(out_val_data_label):
        os.makedirs(out_val_data_label)
    if not os.path.exists(out_test_data_img):
        os.makedirs(out_test_data_img)
    if not os.path.exists(out_test_data_label):
        os.makedirs(out_test_data_label)

    # 将dota数据划分训练验证以及测试集
    split_dota2traindata(input_img_path, input_label_path,
                         out_train_data_img, out_train_data_label, out_val_data_img, out_val_data_label,
                         out_test_data_img, out_test_data_label)
