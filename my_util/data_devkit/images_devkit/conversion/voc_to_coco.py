#!/usr/bin/python

# pip install lxml

import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = None


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def copy_voc2coco(main_label_name, input_voc_img_path, input_voc_label_oath, output_coco_img_path, output_temp_label_path):
    with open(main_label_name, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            # 将影像文件拷贝到coco的影像文件夹中
            img_name = eachline[0] + '.jpg'
            input_voc_img = os.path.join(input_voc_img_path, img_name)
            output_coco_img= os.path.join(output_coco_img_path, img_name)
            shutil.copyfile(input_voc_img, output_coco_img)
            # 将标签文件拷贝到temp文件夹中
            label_name = eachline[0] + '.xml'
            input_voc_label = os.path.join(input_voc_label_oath, label_name)
            output_temp_label = os.path.join(output_temp_label_path, label_name)
            shutil.copyfile(input_voc_label, output_temp_label)
            line = f.readline()


if __name__ == "__main__":

    # 传入VOC文件夹，按照main中的txt将图像分别拷贝到coco图像数据中，然后分别写入两个临时文件夹，分别传入convert，生成json
    voc_path = '/home/data/windowdata/data/detectionDataset/VOC2007/VOC2007'
    temp_path = '/home/data/windowdata/data/detectionDataset/VOC2007/temp'
    coco_path = '/home/data/windowdata/data/detectionDataset/VOC2007/voc2007_coco'
    # voc
    voc_img_path = os.path.join(voc_path, "JPEGImages")
    # 如果利用组件中的VOC 需要修改voc_img_path JPEGImages改为Images
    voc_label_path = os.path.join(voc_path, "Annotations")
    voc_trainval_label_name = os.path.join(voc_path, "ImageSets", "Main", 'trainval.txt')
    voc_test_label_name = os.path.join(voc_path, "ImageSets", "Main", "test.txt")
    # temp
    temp_train_path = os.path.join(temp_path, "Annotations")
    temp_val_path = os.path.join(temp_path, "Annotations")
    # coco
    coco_train_img_path = os.path.join(coco_path, "train")
    coco_val_img_path = os.path.join(coco_path, "test")
    coco_annotations_path = os.path.join(coco_path, "annotations")
    coco_train_json = os.path.join(coco_annotations_path, "instances_train.json")
    coco_val_json = os.path.join(coco_annotations_path, "instances_val.json")

    if not os.path.exists(temp_train_path):
        os.makedirs(temp_train_path)
    if not os.path.exists(temp_val_path):
        os.makedirs(temp_val_path)
    if not os.path.exists(coco_train_img_path):
        os.makedirs(coco_train_img_path)
    if not os.path.exists(coco_val_img_path):
        os.makedirs(coco_val_img_path)
    if not os.path.exists(coco_annotations_path):
        os.makedirs(coco_annotations_path)

    # 通过voc中main文件夹中的信息将voc中img文件拷贝到coco的img文件夹，将voc中标签拷贝到temp_label文件夹
    # train
    copy_voc2coco(voc_trainval_label_name, voc_img_path, voc_label_path, coco_train_img_path, temp_train_path)
    # # test
    copy_voc2coco(voc_test_label_name, voc_img_path, voc_label_path, coco_val_img_path, temp_val_path)

    # 生成coco的json标签文件
    temp_train_files = glob.glob(os.path.join(temp_train_path, "*.xml"))
    temp_val_files = glob.glob(os.path.join(temp_val_path, "*.xml"))
    print("Number of train xml files: {}".format(len(temp_train_files)))
    print("Number of test xml files: {}".format(len(temp_val_files)))
    convert(temp_train_files, coco_train_json)
    convert(temp_val_files, coco_val_json)
