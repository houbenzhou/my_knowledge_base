import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_images(path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pic_names = os.listdir(path)

    for pic_name in pic_names:
        voc_pic_name = pic_name.split('.')
        if voc_pic_name[-1] == "tif":
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
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/data1/test/images",
        help="Base path for dota data",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/dota_format/test/images',
        help="Output base path for dota data",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_dota_path = args.input_dota_path
    out_path = args.out_path

    # 生成DOTA的图像数据
    create_images(input_dota_path, out_path)
