import argparse
import os
import shutil

from PIL import Image, ImageDraw


def visual_dota(img_path, label_path, out_path):
    """
    可视化voc数据集
    :param img_path:
    :param label_path:
    :param out_path:
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
        print(label_name)
        while True:
            mystr = file.readline()  # 表示一次读取一行
            if not mystr:
                # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                break

            label_file_list = mystr.split(' ')
            xmin = float(label_file_list[3])
            ymin = float(label_file_list[4])
            xmax = float(label_file_list[5])
            ymax = float(label_file_list[6])
            categoty = str(label_file_list[0])

            draw.line((xmin, ymin, xmax, ymin), fill=(255, 0, 0), width=4)
            draw.line((xmax, ymin, xmax, ymax), fill=(255, 0, 0), width=4)
            draw.line((xmax, ymax, xmin, ymax), fill=(255, 0, 0), width=4)
            draw.line((xmin, ymax, xmin, ymin), fill=(255, 0, 0), width=4)
            draw.text((xmin, ymin - 44), categoty, fill="#0000ff")
        img.save(os.path.join(out_path, img_name))


def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--img_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images",
        help="image data path",
    )

    parser.add_argument(
        "--label_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/labelTxt',
        help="label path ",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/windowdata/temp/visual_dota_test',
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
        os.makedirs(out_path)


    visual_dota(img_path, label_path, out_path)
