import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_annotation(path_label, dota_labels_path):
    label_names = os.listdir(path_label)
    if not os.path.exists(dota_labels_path):
        os.makedirs(dota_labels_path)

    for label_name in label_names:
        anno_file = os.path.join(path_label, label_name)
        out_file = os.path.join(dota_labels_path, label_name)

        with open(out_file, 'a') as file_out:

            tree_objects = ET.parse(anno_file)
            tree = tree_objects.find("objects")
            list_point = []
            outline = 'imagesource:GoogleEarth'
            file_out.write(outline + '\n')
            outline = 'gsd:1'
            file_out.write(outline + '\n')
            for obj in tree.findall("object"):
                cls = obj.find("possibleresult").find('name').text
                bboxes = obj.find("points").findall("point")
                for bbox in bboxes:
                    list_point.append(bbox.text.split(',')[0])
                    list_point.append(bbox.text.split(',')[1])
                outline = str(list_point[0]) + ' ' + str(list_point[1]) + ' ' + str(list_point[2]) + ' ' + str(
                    list_point[3]) + ' ' + str(list_point[4]) + ' ' + str(list_point[5]) + ' ' + str(
                    list_point[6]) + ' ' + str(list_point[7]) + ' ' + cls + ' ' + str(0)
                file_out.write(outline + '\n')


def create_images(path, voc_labels_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    label_names = os.listdir(voc_labels_path)

    for label_name in label_names:

        pic_name = label_name.split('.')
        if pic_name[-1] == "xml":
            pic_name[-1] = 'tif'
            pic_name = str.join(".", pic_name)
        voc_pic_name = label_name.split('.')
        if voc_pic_name[-1] == "xml":
            voc_pic_name[-1] = 'png'
            voc_pic_name = str.join(".", voc_pic_name)
        im = Image.open(os.path.join(path, pic_name))
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im)

        bg.save(out_path + "/" + voc_pic_name)


def get_parser():
    parser = argparse.ArgumentParser(description="split dota")
    parser.add_argument(
        "--input_dota_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/data/train",
        help="Base path for dota data",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/dota_format/train',
        help="Output base path for dota data",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_dota_path = args.input_dota_path
    out_path = args.out_path
    path_images = os.path.join(input_dota_path, "images")
    path_label = os.path.join(input_dota_path, "label_xml")
    dota_labels_path = os.path.join(out_path, "labelTxt")
    dota_images_path = os.path.join(out_path, "images")

    # 生成DOTA的标签数据
    create_annotation(path_label, dota_labels_path)
    # 生成DOTA的图像数据
    create_images(path_images, dota_labels_path, dota_images_path)
