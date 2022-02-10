# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license:
@contact: houbenzhou@buaa.edu.cn
@software:
@desc:
"""
import cv2
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

import yaml
from dotmap import DotMap

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
    return classes

"""
    使用OpenCV截取图片
"""

def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = cv2.imread(path)

    print("This image's size: {}".format(img.shape))  # (H, W, C)

    plt.figure("Image Contrast")
    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)  # 展示图片的颜色会改变
    plt.axis('off')

    cropped = img[upper:lower, left:right]

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(cropped)
    plt.axis('off')
    plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = cv2.imread(path)  # 打开图像
    cropped = img[upper:lower, left:right]

    # 保存截取的图片
    cv2.imwrite(save_path, cropped)





def extract_categories_from_voc(train_data_path, out_data, class_names=[]):
    """
         从voc数据集中截取
     :param path: 图片路径
     :param left: 区块左上角位置的像素点离图片左边界的距离
     :param upper：区块左上角位置的像素点离图片上边界的距离
     :param right：区块右下角位置的像素点离图片左边界的距离
     :param lower：区块右下角位置的像素点离图片上边界的距离
      故需满足：lower > upper、right > left
     :param save_path: 所截图片保存位置
     """

    source_images_path=os.path.join(train_data_path, "Images")
    source_label_path = os.path.join(train_data_path, "Annotations")
    source_images_filenames = os.listdir(source_images_path)
    class_names = []
    for source_images_filename in source_images_filenames:
        fileid=os.path.splitext(source_images_filename)[0]
        anno_file = os.path.join(source_label_path, fileid + ".xml")

        tree = ET.parse(anno_file)
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            category = cls
            # 获取数据集中category字段保存的类别
            if category not in class_names:
                class_names.append(category)


    for i in class_names:
        if not os.path.exists(os.path.join(out_data,i)):
            os.makedirs(os.path.join(out_data,i))

    for source_images_filename in source_images_filenames:
        fileid=os.path.splitext(source_images_filename)[0]
        anno_file = os.path.join(source_label_path, fileid + ".xml")
        jpeg_file = os.path.join(source_images_path, source_images_filename)

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        i=0
        for obj in tree.findall("object"):
            i=i+1
            cls = obj.find("name").text

            bbox = obj.find("bndbox")
            bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

            image_cut_save(jpeg_file, bbox[0], bbox[1], bbox[2], bbox[3], os.path.join(out_data,cls,fileid+'_'+str(i)+'.jpg'))



if __name__ == '__main__':
    train_data_path = r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_voc"

    out_data =  r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\small_pic"

    class_names = get_classname(train_data_path)

    extract_categories_from_voc(train_data_path, out_data, class_names)
