# !/usr/bin/env python3
# coding=utf-8
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from fvcore.common.file_io import PathManager

from .data import DatasetCatalog, MetadataCatalog
from .structures import BoxMode


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
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)

            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_iobjectspy_voc(name, train_data_path, split, class_names):
    data_path_name = train_data_path.split("/")[-1]
    DatasetCatalog.register(name, lambda: load_voc_instances(train_data_path, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=class_names, dirname=data_path_name, split=split
    )


def register_all_pascal_voc(train_data_path="datasets", class_names=None):
    if class_names is None:
        class_names = ["unspecified"]
    data_path_name = train_data_path.split("/")[-1]
    SPLITS = [
        (data_path_name + "_trainval", "iobjectspy_voc", "trainval"),
        (data_path_name + "_train", "iobjectspy_voc", "train"),
        (data_path_name + "_val", "iobjectspy_voc", "val"),
        (data_path_name + "_test", "iobjectspy_voc", "test"),
    ]
    for name, dirname, split in SPLITS:
        register_iobjectspy_voc(name, train_data_path, split, class_names)
        MetadataCatalog.get(name).evaluator_type = "iobjectspy"


from dotmap import DotMap


def get_classname(train_data_path):
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        voc_config = DotMap(config_dict)
        classes = voc_config.dataset.get('classes')
        del (classes[0])
    return classes


def get_class_num(train_data_path):
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        voc_config = DotMap(config_dict)
        classes = voc_config.dataset.get('classes')
        num = len(classes) - 1
    return num
