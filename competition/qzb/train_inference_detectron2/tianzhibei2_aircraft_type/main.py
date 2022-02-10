# !/usr/bin/env python3
# coding=utf-8

import os
import shutil
import time
import warnings
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

warnings.filterwarnings('ignore', category=FutureWarning)
from iobjectspy import open_datasource
from iobjectspy.ml.vision import DataPreparation, PictureTrainer, PictureInference,ImageryInference

curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.join(curr_dir)
"""
    基于cascade rcnn的图片目标检测工作流
"""
def visual_detetron2(voc_img, voc_xml, out_path):
        xml_path_ = os.listdir(voc_xml)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for xml_name in xml_path_:
            xml_pth = os.path.join(voc_xml, xml_name)
            img_name = str(xml_name.replace('xml', 'jpg'))
            img_pth = os.path.join(voc_img, img_name)

            img = Image.open(img_pth)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            tree = ET.parse(xml_pth)
            rect = {}
            root = tree.getroot()
            for ob in root.iter('object'):
                for name in ob.iter('name'):
                    rect['name'] = name.text
                for score in ob.iter('score'):
                    rect['score'] = score.text
                for bndbox in ob.iter('bndbox'):
                    for xmin in bndbox.iter('xmin'):
                        rect['xmin'] = float(xmin.text)
                    for ymin in bndbox.iter('ymin'):
                        rect['ymin'] = float(ymin.text)
                    for xmax in bndbox.iter('xmax'):
                        rect['xmax'] = float(xmax.text)
                    for ymax in bndbox.iter('ymax'):
                        rect['ymax'] = float(ymax.text)
                    draw.line((rect['xmin'], rect['ymin'], rect['xmax'], rect['ymin']), fill=(255, 0, 0), width=4)
                    draw.line((rect['xmax'], rect['ymin'], rect['xmax'], rect['ymax']), fill=(255, 0, 0), width=4)
                    draw.line((rect['xmax'], rect['ymax'], rect['xmin'], rect['ymax']), fill=(255, 0, 0), width=4)
                    draw.line((rect['xmin'], rect['ymax'], rect['xmin'], rect['ymin']), fill=(255, 0, 0), width=4)
                    draw.text((rect['xmin'], rect['ymin'] ), rect['name']+" "+rect['score'], fill="#0000ff")
            img.save(os.path.join(out_path, img_name))


def example_predict():
    image_path = '/home/data1/competition/data/all_aircraft/qinagzhibei_test'
    # image_names = os.listdir(image_path)
    model_path = os.path.join(curr_dir, '..', 'out', 'tzb2_airplane_voc_v0_20210803', 'baseline', 'model', 'save_model',
                              'save_model.sdm')

    out_data = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'out_result_data')
    out_visual_data = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'out_visual_data')
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    if not os.path.exists(out_visual_data):
        os.makedirs(out_visual_data)
    pic_inference=PictureInference(model_path,out_format='tzb2')
    pic_inference.object_detect_infer(input_data=image_path, out_data=out_data,
              out_dataset_name='out_plane',
        category_name=None, nms_thresh=0.3, score_thresh=0.3)
    visual_detetron2(image_path, out_data, out_visual_data)


if __name__ == '__main__':


    example_predict()



