import argparse
import os

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_parser():
    parser = argparse.ArgumentParser(description="split dota")
    parser.add_argument(
        "--input_ppm_path",
        default="/home/data/windowdata/data/traffic_sign/GTSDB/TrainIJCNN2013",
        help="Base path for ppm data",
    )

    parser.add_argument(
        "--out_jpg_path",
        default='/home/data/windowdata/data/traffic_sign/GTSDB/TrainIJCNN2013_JPG',
        help="Output base path for dota data",
    )

    return parser


def ppm_to_jpg(input_ppm_path, out_jpg_path):
    if not os.path.exists(out_jpg_path):
        os.makedirs(out_jpg_path)
    ppm_names = os.listdir(input_ppm_path)

    for ppm_name in ppm_names:
        # if os.path.isfile(os.path.join(input_ppm_path,ppm_name):
        ppm_path = os.path.join(input_ppm_path, ppm_name)
        if os.path.join(input_ppm_path, ppm_name).endswith(('jpg', 'png', 'jpeg', 'ppm')):
            pic_name = ppm_name.split('.')
            if pic_name[-1] == "ppm":
                pic_name[-1] = 'jpg'
                pic_name = str.join(".", pic_name)
            im = Image.open(ppm_path)
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im)

            bg.save(out_jpg_path + "/" + pic_name)


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_ppm_path = args.input_ppm_path
    out_jpg_path = args.out_jpg_path

    ppm_to_jpg(input_ppm_path, out_jpg_path)
