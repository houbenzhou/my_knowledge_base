# 特征框集合
import codecs
import json
import os
from collections import OrderedDict
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
from iobjectspy.ml.toolkit._toolkit import save_config_to_yaml
import xml.etree.ElementTree as ET
ImageFile.LOAD_TRUNCATED_IMAGES = True

def casia_to_voc(source_images, source_label, out_data):
    """
    方法将原路径中后缀格式为prefix的文件拷贝到out_path中

    :param source_images: 天智杯2影像文件
    :type source_images: str
    :param source_label:  天智杯2图像文件
    :type source_label: str
    :param out_data: 后缀
    :type out_data: str

    """
    source_images_filenames=os.listdir(source_images)
    out_images_path = os.path.join(out_data, 'Images')
    out_label_path = os.path.join(out_data, 'Annotations')
    out_main_path = os.path.join(out_data, "ImageSets", "Main")
    sda_path = os.path.join(out_data, os.path.basename(out_data)+'.sda')

    tile_size=0
    tile_size_offset=0
    categorys_temp = []
    if not os.path.exists(out_images_path):
        os.makedirs(out_images_path)
    if not os.path.exists(out_label_path):
        os.makedirs(out_label_path)
    if not os.path.exists(out_main_path):
        os.makedirs(out_main_path)
    for source_images_filename in source_images_filenames:
        temp_source_images_filename_tuple=os.path.splitext(source_images_filename)
        source_label_filename=temp_source_images_filename_tuple[0]+".xml"
        source_images_file = os.path.join(source_images, source_images_filename)
        source_label_file= os.path.join(source_label, source_label_filename)

        out_images_filename = temp_source_images_filename_tuple[0] + ".jpg"
        out_label_filename = temp_source_images_filename_tuple[0] + ".xml"
        out_images_file = os.path.join(out_images_path, out_images_filename)
        out_label_file = os.path.join(out_label_path, out_label_filename)

        # json to xml
        tree = ET.parse(source_label_file)
        # header = s[:2]
        objects = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text

            bbox = obj.find("bndbox")
            bbox_temp = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # bbox = [bbox_temp]
            # center = sum(bbox[0::2]) / 4.0, sum(bbox[1::2]) / 4.0
            objects.append({'bbox': bbox_temp,
                            'label': cls,
                            'difficulty': 0})
        tile_size=400
        tile_size_offset=200
        # source_images_file
        img = cv2.imread(source_images_file)
        width,height,channels=img.shape
        with codecs.open(out_label_file, "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC' + '</folder>\n')
            xml.write('\t<filename>' + temp_source_images_filename_tuple[0] + ".jpg" + '</filename>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(width) + '</width>\n')
            xml.write('\t\t<height>' + str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for multi in objects:
                category = multi["label"]
                # 获取数据集中category字段保存的类别
                if (category == "F/A-18"):
                    category = "F-A-18"
                elif (category == 'F/A-18E/F'):
                    category = "F-A-18E-F"
                elif (category == 'RC-135V/W'):
                    category = "RC-135V-W"
                if category not in categorys_temp:
                    categorys_temp.append(category)

                points = np.array(multi["bbox"])
                xmin = points[0]
                xmax = points[2]
                ymin = points[1]
                ymax = points[3]
                label = category

                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + label + '</name>\n')
                    xml.write('\t\t<modify_name>' + 'normal' + '</modify_name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(round(xmin))) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(round(ymin))) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(round(xmax))) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(round(ymax))) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')

            xml.write('</annotation>')
        # copy images file
        # copyfile(source_images_file, out_images_file)
        # im = Image.open(source_images_file)
        # bg = Image.new("RGB", im.size, (255, 255, 255))
        # bg.paste(im)
        # bg.save(out_images_file)

    # 保存sda文件
    # 1024 更新tile记录数
    dic_voc_yml = OrderedDict({
        'dataset': OrderedDict({"name": "example_voc",
                                'classes': categorys_temp,
                                'image_count': 0,
                                "data_type": "voc",
                                "input_bandnum": 3, "input_ext": 'tif',
                                "x_ext": 'jpg',
                                "tile_size_x": tile_size,
                                "tile_size_y": tile_size,
                                "tile_offset_x": tile_size_offset,
                                "tile_offset_y": tile_size_offset,
                                "image_mean": [115.6545965, 117.62014299, 106.01483799],
                                "image_std": [56.82521775, 53.46318049, 56.07113724]}),

    })
    save_config_to_yaml(dic_voc_yml, sda_path)
    # 保存文件索引
    _save_index_file(out_main_path, out_label_path)


def _save_index_file(output_path_main, output_path_img):
    if not os.path.exists(output_path_main):
        os.makedirs(output_path_main)
    # 随机将数据分为train、val、test数据
    pic_names = os.listdir(output_path_img)
    # 分配训练数据验证数据的数组长度
    train_length = int((len(pic_names) / 5) * 4)
    val_length = int(len(pic_names) / 10)
    # 训练数据集、验证数据集、测试数据集数组
    list_train = pic_names[0:train_length]
    list_val = pic_names[train_length:train_length + val_length]
    list_test = pic_names[train_length:]
    list_trainval = list_train + list_val
    # 打开创建的文件
    train_txt = open(os.path.join(output_path_main, 'train.txt'), "w")
    val_txt = open(os.path.join(output_path_main, 'val.txt'), "w")
    test_txt = open(os.path.join(output_path_main, 'test.txt'), "w")
    trainval_txt = open(os.path.join(output_path_main, 'trainval.txt'), "w")

    for pic_name in list_train:
        label_name = os.path.splitext(pic_name)[0]
        train_txt.write(label_name + '\n')
    for pic_name in list_val:
        label_name = os.path.splitext(pic_name)[0]
        val_txt.write(label_name + '\n')
    for pic_name in list_test:
        label_name = os.path.splitext(pic_name)[0]
        test_txt.write(label_name + '\n')
    for pic_name in list_trainval:
        label_name = os.path.splitext(pic_name)[0]
        trainval_txt.write(label_name + '\n')

    # 关闭所有打开的文件
    train_txt.close()
    val_txt.close()
    test_txt.close()
    trainval_txt.close()



if __name__ == '__main__':
    source_images = r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_all\img'
    source_label = r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_all\label'
    out_data = r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_voc'

    casia_to_voc(source_images, source_label, out_data)












