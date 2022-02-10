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


def voc_xml_to_xml_get_category_name(input_voc_xml, out_voc_xml, category_list):
    xml_path_ = os.listdir(input_voc_xml)
    xml_name_list = []
    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_pth = os.path.join(input_voc_xml, xml_name)
        tree = ET.parse(xml_pth)
        root = tree.getroot()
        cp_file = False
        for ob in root.iter('object'):
            for name in ob.iter('name'):
                label = name.text
                if label in category_list:
                    cp_file = True
        if cp_file:
            xml_name_list.append(xml_name)

    for xml_name in xml_name_list:
        input_file = os.path.join(input_voc_xml, xml_name)
        out_file = os.path.join(out_voc_xml, xml_name)
        copyfile(input_file, out_file)

def copy_labelname_images_to_outimages(input_images, output_label, output_images):
    output_label_name_list = os.listdir(output_label)
    for output_label_name in output_label_name_list:
        output_name=os.path.splitext(output_label_name)[0]
        output_images_name=os.path.join(output_name+'.jpg')
        input_file = os.path.join(input_images, output_images_name)
        out_file = os.path.join(output_images, output_images_name)
        copyfile(input_file, out_file)

def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--input_data",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\modify_casia_2_qzb_voc",
        help="voc image path",
    )
    parser.add_argument(
        "--output_data",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_08_form_casia",
        help="voc label path",
    )
    parser.add_argument(
        "--label_dict",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_voc\New_Annotations",
        help="voc label path",
    )

    return parser

if __name__ == '__main__':

    args = get_parser().parse_args()
    input_data = args.input_data
    output_data = args.output_data
    input_label=os.path.join(input_data,'Annotations')
    input_images=os.path.join(input_data,'Images')
    output_label=os.path.join(output_data,'Annotations')
    output_images=os.path.join(output_data,'Images')

    category_list=['01','08']
    if not os.path.exists(output_label):
        os.makedirs(output_label)
    if not os.path.exists(output_images):
        os.makedirs(output_images)
    voc_xml_to_xml_get_category_name(input_label, output_label,category_list)
    copy_labelname_images_to_outimages(input_images,output_label,output_images)
    source_images = os.path.join(output_data, 'Images')
    source_label = os.path.join(output_data, 'Annotations')
    out_main_path = os.path.join(output_data, "ImageSets", "Main")
    sda_file = os.path.join(output_data, os.path.basename(output_data) + '.sda')
    _save_sda_file(source_images, source_label, sda_file, tile_shape=(399, 399))
    _save_index_file(out_main_path, source_images)
