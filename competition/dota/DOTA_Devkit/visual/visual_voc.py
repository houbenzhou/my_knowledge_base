import argparse
import os
import shutil
import xml.etree.ElementTree as ET

## 评估图片目标的长宽比
from PIL import Image, ImageDraw


def eval_objects_width_height(xml_paths, object_width_height_txt):
    xml_path_ = os.listdir(xml_paths)
    for pic_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_path = os.path.join(xml_paths, pic_name)
        tree = ET.parse(xml_path)
        rect = {}
        root = tree.getroot()
        for name in root.iter('path'):
            rect['path'] = name.text
        for difficult in ob.iter('difficult'):
            rect['difficult'] = int(difficult.text)
        for ob in root.iter('object'):
            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = xmin.text
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = ymin.text
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = xmax.text
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = ymax.text
                # 判断长边与短边的比例
                # if ((float(rect['ymax']) - float(rect['ymin'])) / (
                #         float(rect['xmax']) - float(rect['xmin'])) <= 0.1) | (
                #         (float(rect['ymax']) - float(rect['ymin'])) / (
                #         float(rect['xmax']) - float(rect['xmin'])) >= 10):
                #     line = "xmin:" + str(rect['xmin']) + "ymin:" + str(rect['ymin']) + "xmax:" + str(
                #         rect['xmax']) + "ymax:" + str(rect['ymax']) + "y/x:" + str(
                #         (float(rect['ymax']) - float(rect['ymin'])) / (float(rect['xmax']) - float(rect['xmin'])))
                #     object_width_height_txt.write(line + '\n')
                #
            line = "图片:" + "y/x:" + str(
                (float(rect['ymax']) - float(rect['ymin'])) / (float(rect['xmax']) - float(rect['xmin'])))
            object_width_height_txt.write(line + '\n')


def visual_object_detection_voc(voc_img, voc_xml, out_path):
    # voc_xml = os.path.join(voc_path, 'Annotations')
    # voc_img = os.path.join(voc_path, 'Images')
    xml_path_ = os.listdir(voc_xml)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
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
                draw.text((rect['xmin'], rect['ymin'] - 44), rect['name'], fill="#0000ff")
        img.save(os.path.join(out_path, img_name))


def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--voc_img",
        default="/home/data/hou/workspaces/iobjectspy3/resources_ml/out/voc/Images",
        help="voc image path",
    )
    parser.add_argument(
        "--voc_xml",
        default="/home/data/hou/workspaces/iobjectspy3/resources_ml/out/plane",
        help="voc label path",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/windowdata/temp/dota_train_data_visual/dotav1/trainsplite800_voc',
        help="A directory to save the output images . ",
    )

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    voc_img = args.voc_img
    voc_xml = args.voc_xml
    out_path = args.out_path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    visual_object_detection_voc(voc_img, voc_xml, out_path)
