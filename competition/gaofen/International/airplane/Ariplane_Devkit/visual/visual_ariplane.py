import argparse
import os
import shutil
import xml.etree.ElementTree as ET

## 评估图片目标的长宽比
from PIL import Image, ImageDraw


def visual_object_detection_voc(img_path, xml_path, out_path):
    # voc_xml = os.path.join(voc_path, 'Annotations')
    # voc_img = os.path.join(voc_path, 'Images')
    xml_path_ = os.listdir(xml_path)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_pth = os.path.join(xml_path, xml_name)
        img_name = str(xml_name.replace('xml', 'tif'))
        img_pth = os.path.join(img_path, img_name)

        img = Image.open(img_pth)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        tree = ET.parse(xml_pth)
        rect = {}
        root = tree.getroot()
        for ob in root.iter('object'):
            for name in ob.iter('name'):
                rect['name'] = name.text
                rect['point'] = ""

            for bndbox in ob.iter('points'):
                for point in bndbox.iter('point'):
                    rect['point'] = rect['point'] + '_' + str(point.text)
                points = rect['point'].split('_')
                x1 = float(points[1].split(',')[0])
                y1 = float(points[1].split(',')[1])
                x2 = float(points[2].split(',')[0])
                y2 = float(points[2].split(',')[1])
                x3 = float(points[3].split(',')[0])
                y3 = float(points[3].split(',')[1])
                x4 = float(points[4].split(',')[0])
                y4 = float(points[4].split(',')[1])
                x5 = float(points[5].split(',')[0])
                y5 = float(points[5].split(',')[1])

                draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=4)
                draw.line((x2, y2, x3, y3), fill=(255, 0, 0), width=4)
                draw.line((x3, y3, x4, y4), fill=(255, 0, 0), width=4)
                draw.line((x4, y4, x5, y5), fill=(255, 0, 0), width=4)
                # draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=4)
                # draw.line((rect['xmax'], rect['ymin'], rect['xmax'], rect['ymax']), fill=(255, 0, 0), width=4)
                # draw.line((rect['xmax'], rect['ymax'], rect['xmin'], rect['ymax']), fill=(255, 0, 0), width=4)
                # draw.line((rect['xmin'], rect['ymax'], rect['xmin'], rect['ymin']), fill=(255, 0, 0), width=4)
                draw.text((x1, y1 - 44), rect['name'], fill="#0000ff")
        img.save(os.path.join(out_path, img_name))


def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--img",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/data1/test/images",
        help="voc image path",
    )
    parser.add_argument(
        "--xml",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/dota_format/test/result/submission2",
        help="voc label path",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/windowdata/temp/1',
        help="A directory to save the output images . ",
    )

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    img = args.img
    xml = args.xml
    out_path = args.out_path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    visual_object_detection_voc(img, xml, out_path)
