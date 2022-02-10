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
from iobjectspy.ml.vision import DataPreparation, PictureTrainer, PictureInference

curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.join(curr_dir,'..')
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

def example_create_train_data():
    """
        训练数据生成
    """
    input_data = os.path.join(curr_dir, '..', 'example_data', 'training', 'plane.tif')
    ds = open_datasource(os.path.join(curr_dir, '..', 'example_data', 'training', 'label.udbx'))
    input_label = ds['label']
    label_class_field = None
    output_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane')
    output_name = 'VOC'
    tile_size = 800
    tile_offset = 400
    output_data_path = os.path.join(output_path, output_name)
    if os.path.exists(output_data_path):
        shutil.rmtree(output_data_path)
    start_time = time.time()

    DataPreparation.create_training_data(input_data=input_data, input_label=input_label,
                                         label_class_field=label_class_field, output_path=output_path,
                                         output_name=output_name,
                                         training_data_format='VOC', tile_format='jpg', tile_size_x=tile_size,
                                         tile_size_y=tile_size, tile_offset_x=tile_offset,
                                         tile_offset_y=tile_offset, tile_start_index=0, save_nolabel_tiles=False)

    print('完成，共耗时{}s，训练数据保存在 {}'.format(
        time.time() - start_time, os.path.join(output_path, output_name)))

def example_train():
    train_data_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane','VOC')
    config = os.path.join(curr_dir, '..', 'trainer_config','object_detection', 'train_config_cascade_rcnn_R_50_FPN_3x.sdt')
    backbone_weight_path = os.path.join(curr_dir, '..', 'backbone', 'R-50.pkl')

    log_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'log')
    output_model_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'model')
    output_model_name = "save_model"
    lr= 0.02
    PictureTrainer(train_data_path=train_data_path, config=config, epoch=20, batch_size=2, lr=lr,
            output_model_path=output_model_path,
            output_model_name=output_model_name, backbone_name='renet50',
            backbone_weight_path=backbone_weight_path, log_path=log_path, reload_model=False).object_detect_train()



def example_predict_pic():
    image_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'VOC', 'Images')
    model_path = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'model', 'save_model', 'save_model.sdm')

    out_data = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'out_result_data')
    out_visual_data = os.path.join(curr_dir, '..', 'out', 'picture_cascade_rcnn_plane', 'out_visual_data')
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    if not os.path.exists(out_visual_data):
        os.makedirs(out_visual_data)
    pic_inference=PictureInference(model_path)
    pic_inference.object_detect_infer(input_data=image_path, out_data=out_data,
              out_dataset_name='out_plane',
        category_name=None, nms_thresh=0.3, score_thresh=0.3)
    visual_detetron2(image_path, out_data, out_visual_data)

if __name__ == '__main__':
    example_create_train_data()
    example_train()
    example_predict_pic()

