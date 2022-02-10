import json, codecs, os
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

## 评估图片目标的长宽比
from PIL import Image, ImageDraw
from PIL.ImageFile import ImageFile

from resources_ml.competition.data_devkit.voc_devkit.create_sda_ImageSets_from_xml_images import _save_sda_file, \
    _save_index_file

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from  shutil import copyfile

def copy_file_path(source_path, out_path,prefix=None):
    """
    方法将原路径中后缀格式为prefix的文件拷贝到out_path中

    :param source_path: 原文件路径
    :type source_path: str
    :param out_path:  输出文件路径
    :type out_path: str
    :param prefix: 后缀
    :type prefix: str

    """
    source_filenames=os.listdir(source_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for source_filename in source_filenames:
        if prefix is None:
            source_file_path = os.path.join(source_path, source_filename)
            out_file_path = os.path.join(out_path, source_filename)
            copyfile(source_file_path, out_file_path)
        elif source_filename.endswith(prefix):
            source_file_path = os.path.join(source_path, source_filename)
            out_file_path=os.path.join(out_path, source_filename)
            copyfile(source_file_path, out_file_path)

def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--input_data_1",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\tzb2_airplane_voc_modify_v2_20210806\voc_600",
        help="voc image path",
    )
    parser.add_argument(
        "--input_data_2",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_08_form_casia",
        help="voc image path",
    )
    parser.add_argument(
        "--output_data",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_08_form_tzb2",
        help="voc label path",
    )


    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    input_data_1 = args.input_data_1
    input_data_2 = args.input_data_2
    output_data = args.output_data
    input_label_1=os.path.join(input_data_1,'Annotations')
    input_images_1=os.path.join(input_data_1,'Images')
    input_label_2=os.path.join(input_data_2,'Annotations')
    input_images_2=os.path.join(input_data_2,'Images')
    output_label=os.path.join(output_data,'Annotations')
    output_images=os.path.join(output_data,'Images')

    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_images):
        os.makedirs(output_images)
    copy_file_path(input_label_1, output_label,'xml')
    copy_file_path(input_label_2, output_label,'xml')
    copy_file_path(input_images_1, output_images,'jpg')
    copy_file_path(input_images_2, output_images,'jpg')
    source_images = os.path.join(output_data, 'Images')
    source_label = os.path.join(output_data, 'Annotations')
    out_main_path = os.path.join(output_data, "ImageSets", "Main")
    sda_file = os.path.join(output_data, os.path.basename(output_data) + '.sda')
    _save_sda_file(source_images, source_label, sda_file, tile_shape=(399, 399))
    _save_index_file(out_main_path, source_images)
