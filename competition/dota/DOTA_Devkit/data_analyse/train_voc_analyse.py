# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import itertools
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from dotmap import DotMap
from fvcore.common.file_io import PathManager
from tabulate import tabulate


def load_voc_instances(dirname: str, split: str, class_names: list):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "Images", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            # 统计difficult==1的目标数量
            # 统计目标数量
            # 统计小目标数量，中等目标数量，大目标数量（暂定以Coco的标准为主）
            # 统计目标长宽比分布
            # 利用以上信息给出切图的标准参数，以及anchor的scale以及比率
            # 分数目标检测
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)

            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def _print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)


def _print_instances_small_middle_larg_histogram(dataset_dicts, class_names):
    """
     Args:
         dataset_dicts (list[dict]): list of dataset dicts.
         class_names (list[str]): list of class names (zero-indexed).
     """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram_total = np.zeros((num_classes,), dtype=np.int)
    histogram_small = np.zeros((num_classes,), dtype=np.int)
    histogram_medium = np.zeros((num_classes,), dtype=np.int)
    histogram_large = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes_total = [x["category_id"] for x in annos]

        classes_small = [x["category_id"] for x in annos if
                         (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= (32 * 32)]
        classes_medium = [x["category_id"] for x in annos if
                          (32 * 32) <= (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) <= (96 * 96)]
        classes_large = [x["category_id"] for x in annos if
                         (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]) >= (96 * 96)]
        histogram_total += np.histogram(classes_total, bins=hist_bins)[0]
        histogram_small += np.histogram(classes_small, bins=hist_bins)[0]
        histogram_medium += np.histogram(classes_medium, bins=hist_bins)[0]
        histogram_large += np.histogram(classes_large, bins=hist_bins)[0]
    # histogram=histogram+histogram1
    histogram = np.concatenate((histogram_total, histogram_small, histogram_medium, histogram_large)).reshape(4,
                                                                                                              num_classes).T

    # 原始数组转置a.T

    N_COLS = min(5, len(class_names) * 4)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v[0]), int(v[1]), int(v[2]), int(v[3])] for i, v in
                          enumerate(histogram)])
    )
    # total_num_instances = sum(data[1::2])
    total_num_instances=0
    total_small_instances=0
    total_medium_instances=0
    total_large_instances=0
    for i in range(0, num_classes):
        total_num_instances = total_num_instances + data[i * 5 + 1]
        total_small_instances = total_small_instances + data[i * 5 + 2]
        total_medium_instances = total_medium_instances + data[i * 5 + 3]
        total_large_instances = total_large_instances + data[i * 5 + 4]
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    # if num_classes > 1:
    data.extend(["total", total_num_instances, total_small_instances, total_medium_instances, total_large_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instance", "small", "medium", "large"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    print(table)


def get_classname(train_data_path):
    """
   Args:
       train_data_path (str)]): train data path
   """
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        voc_config = DotMap(config_dict)
        classes = voc_config.dataset.get('classes')
        del (classes[0])
    return classes


def conunt_instances_class_number(dataset_dicts, class_names):
    """
   Args:
       dataset_dicts (list[dict]): list of dataset dicts.
       class_names (list[str]): list of class names (zero-indexed).
   """
    for dataset_name, dicts in zip(class_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            _print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass


def count_instances_class_small_middle_large_number(dataset_dicts, class_names):
    """
   Args:
       dataset_dicts (list[dict]): list of dataset dicts.
       class_names (list[str]): list of class names (zero-indexed).
   """
    for dataset_name, dicts in zip(class_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            _print_instances_small_middle_larg_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass


if __name__ == '__main__':
    # train_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/newVoc/VOC"
    train_data_path = "/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC"
    split = "trainval"
    class_names = get_classname(train_data_path)

    dataset_dicts = load_voc_instances(train_data_path, split, class_names)

    # conunt_instances_class_number(dataset_dicts, class_names)

    count_instances_class_small_middle_large_number(dataset_dicts, class_names)
