import argparse
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rasterio
from PIL import Image
from PIL import ImageFile
from rasterio.windows import Window

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_annotation(path_label, dota_labels_path):
    label_names = os.listdir(path_label)
    if not os.path.exists(dota_labels_path):
        os.makedirs(dota_labels_path)

    for label_name in label_names:
        anno_file = os.path.join(path_label, label_name)
        label_name1 = str(label_name.split('.')[0]) + '.txt'

        out_file = os.path.join(dota_labels_path, label_name1)

        with open(out_file, 'w') as file_out:

            tree_objects = ET.parse(anno_file)
            tree = tree_objects.find("objects")

            outline = 'imagesource:GoogleEarth'
            file_out.write(outline + '\n')
            outline = 'gsd:1'
            file_out.write(outline + '\n')
            for obj in tree.findall("object"):
                cls = obj.find("possibleresult").find('name').text
                bboxes = obj.find("points").findall("point")
                list_point = []
                for bbox in bboxes:
                    list_point.append(bbox.text.split(',')[0])
                    list_point.append(bbox.text.split(',')[1])
                # outline = str(list_point[0]) + ' ' + str(list_point[1]) + ' ' + str(list_point[2]) + ' ' + str(
                #     list_point[3]) + ' ' + str(list_point[4]) + ' ' + str(list_point[5]) + ' ' + str(
                #     list_point[6]) + ' ' + str(list_point[7]) + ' ' + cls + ' ' + str(0)
                outline = str(int(float(list_point[0]))) + ' ' + str(int(float(list_point[1]))) + ' ' + str(
                    int(float(list_point[2]))) + ' ' + str(
                    int(float(list_point[3]))) + ' ' + str(int(float(list_point[4]))) + ' ' + str(
                    int(float(list_point[5]))) + ' ' + str(
                    int(float(list_point[6]))) + ' ' + str(int(float(list_point[7]))) + ' ' + cls + ' ' + str(0)
                file_out.write(outline + '\n')


def create_images(path, voc_labels_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    label_names = os.listdir(path)

    for label_name in label_names:

        pic_name = label_name.split('.')
        if pic_name[-1] == "tiff":
            pic_name[-1] = 'tiff'
            pic_name = str.join(".", pic_name)
        voc_pic_name = label_name.split('.')
        if voc_pic_name[-1] == "tiff":
            voc_pic_name[-1] = 'png'
            voc_pic_name = str.join(".", voc_pic_name)
        im = Image.open(os.path.join(path, pic_name))
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im)

        bg.save(out_path + "/" + voc_pic_name)


def reshape_as_image(arr):
    """Returns the source array reshaped into the order
    expected by image processing and visualization software
    (matplotlib, scikit-image, etc)
    by swapping the axes order from (bands, rows, columns)
    to (rows, columns, bands)

    Parameters
    ----------
    arr : array-like of shape (bands, rows, columns)
        image to reshape
    """
    # swap the axes order from (bands, rows, columns) to (rows, columns, bands)
    im = np.ma.transpose(arr, [1, 2, 0])
    return im


def get_label_list(input_label_file, input_label_list):
    tree_objects = ET.parse(input_label_file)
    tree = tree_objects.find("objects")
    for obj in tree.findall("object"):
        cls = obj.find("possibleresult").find('name').text
        bboxes = obj.find("points").findall("point")
        list_point = []
        for bbox in bboxes:
            list_point.append(bbox.text.split(',')[0])
            list_point.append(bbox.text.split(',')[1])
        x = []
        y = []
        x.append(float(list_point[0]))
        x.append(float(list_point[2]))
        x.append(float(list_point[4]))
        x.append(float(list_point[6]))
        y.append(float(list_point[1]))
        y.append(float(list_point[3]))
        y.append(float(list_point[5]))
        y.append(float(list_point[7]))
        xmin = min(x)
        ymin = min(y)
        xmax = max(x)
        ymax = max(y)

        if (ymax > (ymin + 2)) & (xmax > (xmin + 2)):
            if ((ymax - ymin) / (xmax - xmin) <= 9) & ((ymax - ymin) / (xmax - xmin) >= 0.13):
                listnew = []
                listnew.append(cls)
                listnew.append(float(list_point[0]))
                listnew.append(float(list_point[1]))
                listnew.append(float(list_point[2]))
                listnew.append(float(list_point[3]))
                listnew.append(float(list_point[4]))
                listnew.append(float(list_point[5]))
                listnew.append(float(list_point[6]))
                listnew.append(float(list_point[7]))

                listnew.append(0)
                input_label_list.append(listnew)

    return input_label_list


def split_data(input_img_path, input_label_path, split_images_path, split_label_path, blocksize=1024,
               tile_offset=512, suffix_img='tiff', suffix_label="txt"):
    label_names_list = os.listdir(input_label_path)
    for label_name in label_names_list:
        input_label_file = os.path.join(input_label_path, label_name)
        input_label_file = open(input_label_file, "r", encoding="utf-8", errors="ignore")
        input_label_list = []
        input_label_list = get_label_list(input_label_file, input_label_list)

        img_name = label_name.split(".")
        img_name[-1] = suffix_img
        img_name = str.join(".", img_name)
        input_img_file = os.path.join(input_img_path, img_name)
        with rasterio.open(input_img_file) as ds:
            height = ds.height
            width = ds.width
            width_block = (ds.width) // tile_offset
            height_block = (ds.height) // tile_offset
            for i in range(height_block):
                for j in range(width_block):
                    # split image data
                    block_xmin = j * tile_offset
                    block_ymin = i * tile_offset
                    block_xmax = block_xmin + blocksize
                    block_ymax = block_ymin + blocksize
                    if height <= block_ymax:
                        block_ymin = height - blocksize
                        block_ymax = height
                    if width <= block_xmax:
                        block_xmin = width - blocksize
                        block_xmax = width

                    block = np.zeros([3, blocksize, blocksize], dtype=np.uint8)
                    img = ds.read(window=Window(block_xmin, block_ymin, blocksize, blocksize))
                    block[:, :img.shape[1], :img.shape[2]] = img[:3, :, :]
                    block = reshape_as_image(block)
                    block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(split_images_path,
                                             os.path.join(img_name.split(".")[0]) + "_" + str(i) + "_" + str(
                                                 j) + ".png"), block)
                    # split label data
                    with open(os.path.join(split_label_path,
                                           os.path.join(img_name.split(".")[0]) + "_" + str(i) + "_" + str(
                                               j) + ".txt"), 'w') as file_out:
                        for bbox in input_label_list:
                            x = []
                            y = []
                            x.append(float(bbox[1]))
                            x.append(float(bbox[3]))
                            x.append(float(bbox[5]))
                            x.append(float(bbox[7]))
                            y.append(float(bbox[2]))
                            y.append(float(bbox[4]))
                            y.append(float(bbox[6]))
                            y.append(float(bbox[8]))
                            xmin = min(x)
                            ymin = min(y)
                            xmax = max(x)
                            ymax = max(y)
                            difficult = False
                            if (xmin >= block_xmin - 50) & (ymin >= block_ymin - 50) & (xmax <= block_xmax + 50) & (
                                    ymax <= block_ymax + 50):
                                # if xmin < block_xmin:
                                #     difficult = True
                                # if (xmin >= block_xmin) & (ymin >= block_ymin) & (xmax <= block_xmax) & (
                                #         ymax <= block_ymax):
                                outline = str(
                                    int(float(bbox[1] - block_xmin))) + ' ' + str(
                                    int(float(bbox[2] - block_ymin))) + ' ' + str(
                                    int(float(bbox[3] - block_xmin))) + ' ' + str(
                                    int(float(bbox[4] - block_ymin))) + ' ' + str(
                                    int(float(bbox[5] - block_xmin))) + ' ' + str(
                                    int(float(bbox[6] - block_ymin))) + ' ' + str(
                                    int(float(bbox[7] - block_xmin))) + ' ' + str(
                                    int(float(bbox[8] - block_ymin))) + ' ' + bbox[
                                              0] + ' ' + str(0)
                                file_out.write(outline + '\n')


def split_data_to_dota(split_images_path, split_label_path, dota_images_path, dota_labels_path):
    pass


def get_parser():
    parser = argparse.ArgumentParser(description="tianzhibei_sar_to_dota")
    # parser.add_argument(
    #     "--input_img_path",
    #     default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/val/images",
    #     help="input images data path",
    # )
    #
    # parser.add_argument(
    #     "--input_label_path",
    #     default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/val/label_xml",
    #     help="input labels data path",
    # )

    parser.add_argument(
        "--input_img_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/val/images",
        help="input images data path",
    )

    parser.add_argument(
        "--input_label_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/val/label_xml",
        help="input labels data path",
    )

    parser.add_argument(
        "--out_path",
        default='/home/data/windowdata/temp1/test2',
        help="Output base path for dota data",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_img_path = args.input_img_path
    input_label_path = args.input_label_path
    out_path = args.out_path
    out_split_sar_path = os.path.join(out_path, "sar")
    split_images_path = os.path.join(out_path, "images")
    split_label_path = os.path.join(out_path, "labelTxt")
    out_dota_sar_path = os.path.join(out_path, "dota")
    dota_images_path = os.path.join(out_path, "images")
    dota_labels_path = os.path.join(out_path, "labelTxt")
    if not os.path.exists(split_images_path):
        os.makedirs(split_images_path)
    if not os.path.exists(split_label_path):
        os.makedirs(split_label_path)
    if not os.path.exists(dota_images_path):
        os.makedirs(dota_images_path)
    if not os.path.exists(dota_labels_path):
        os.makedirs(dota_labels_path)

    # 将影像数据切分，但保留切分数据的原始信息
    split_data(input_img_path, input_label_path, split_images_path, split_label_path, blocksize=512, tile_offset=256)

    # 将切分的数据转换为dota数据集
    # split_data_to_dota(split_images_path, split_label_path, dota_images_path, dota_labels_path)
