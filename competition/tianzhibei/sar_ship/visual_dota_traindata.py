import argparse
import os
import shutil

from PIL import Image, ImageDraw


def visual_dota(img_path, label_path, out_path):
    """
    可视化voc数据集
    :param input_data
    :param out_path
    :return:
    """
    label_names = os.listdir(label_path)

    for label_name in label_names:
        # 获取图片路径，用于获取图像大小以及通道数
        img_name = label_name.split(".")[0] + '.png'
        img_file = os.path.join(img_path, img_name)
        label_file = os.path.join(label_path, label_name)

        img = Image.open(img_file)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        file = open(label_file, "r", encoding="utf-8", errors="ignore")
        while True:
            mystr = file.readline()  # 表示一次读取一行
            if not mystr:
                # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                break
            label_file_list = mystr.split(' ')
            if len(label_file_list) >= 8:
                x = []
                y = []
                x.append(float(label_file_list[0]))
                x.append(float(label_file_list[2]))
                x.append(float(label_file_list[4]))
                x.append(float(label_file_list[6]))
                y.append(float(label_file_list[1]))
                y.append(float(label_file_list[3]))
                y.append(float(label_file_list[5]))
                y.append(float(label_file_list[7]))
                xmin = min(x)
                ymin = min(y)
                xmax = max(x)
                ymax = max(y)
                categoty = str(label_file_list[8])
                draw.line((xmin, ymin, xmax, ymin), fill=(255, 0, 0), width=4)
                draw.line((xmax, ymin, xmax, ymax), fill=(255, 0, 0), width=4)
                draw.line((xmax, ymax, xmin, ymax), fill=(255, 0, 0), width=4)
                draw.line((xmin, ymax, xmin, ymin), fill=(255, 0, 0), width=4)
                # draw.text((xmin, ymin - 44), categoty, fill="#0000ff")
        img.save(os.path.join(out_path, img_name))

def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--img_path",
        default="/home/data/windowdata/temp1/test2/images",
        help="image data path",
    )

    parser.add_argument(
        "--label_path",
        default='/home/data/windowdata/temp1/test2/labelTxt',
        help="label path ",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/windowdata/temp/1',
        help="A directory to save the output images . ",
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    img_path = args.img_path
    label_path = args.label_path
    out_path = args.out_path
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    visual_dota(img_path, label_path, out_path)
