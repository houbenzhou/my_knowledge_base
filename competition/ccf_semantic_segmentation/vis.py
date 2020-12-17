# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""

import glob
import os
import shutil
import random
from tqdm import tqdm
import time
# import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def vis_image_mask(title, *args):
    l = len(args)
    plt.figure(figsize=(20, 20))
    plt.suptitle(title)
    for i in range(l):
        cur_plt = (1, l, i + 1)
        plt.subplot(*cur_plt)
        plt.imshow(list(args[i].values())[0])
        plt.title(list(args[i].keys())[0])
        # plt.colorbar()
    # plt.savefig(Path + title + ".png")
    plt.show()


def vis_images(image_paths, image_exts, image_titles,random_seed=1100):
    """
    vis multi folder images
    :param image_paths: folder paths-list
    :param image_exts:
    :param image_titles:
    :param random_seed:
    :return:
    """
    images = glob.glob(os.path.join(image_paths[0], '*.' + image_exts[0]))
    random.seed(random_seed)
    random.shuffle(images)
    for im_f in images:
        file_name = os.path.basename(im_f)
        file_name_noext = os.path.splitext(os.path.basename(im_f))[0]
        vis_args = []
        for i in range(len(image_paths)):
            f_path = os.path.join(image_paths[i], file_name_noext + '.' + image_exts[i])
            vis_args.append({image_titles[i]: np.array(Image.open(f_path))})
        vis_image_mask(file_name, *vis_args)


def vis_images_road_water(image_paths, image_exts, image_titles,mask_index=1,random_seed=1999):
    images = glob.glob(os.path.join(image_paths[0], '*.' + image_exts[0]))
    random.seed(random_seed)
    random.shuffle(images)
    ii = 0
    jj = 0
    for im_f in images:
        ii += 1
        file_name = os.path.basename(im_f)
        file_name_noext = os.path.splitext(os.path.basename(im_f))[0]
        vis_args = []
        for i in range(len(image_paths)):
            f_path = os.path.join(image_paths[i], file_name_noext + '.' + image_exts[i])
            vis_args.append({image_titles[i]: np.array(Image.open(f_path))})
        f_path = os.path.join(image_paths[mask_index], file_name_noext + '.' + image_exts[mask_index])
        mask=np.array(Image.open(f_path))
        if np.sum(mask==4)+np.sum(mask==3)>10:
            jj += 1
            mask_road=(mask==4).astype(np.uint8)
            mask_water=(mask==3).astype(np.uint8)
            vis_args.append({image_titles[i]+'water': mask_water})
            vis_args.append({image_titles[i]+'road': mask_road})
            title = "{index1}_{index2}".format(index1=ii, index2=jj)
            vis_image_mask(title, *vis_args)


# 统计_wr中的连通块数目
def statis_road_water(mask_path, ext):
    images = glob.glob(os.path.join(mask_path, '*.' + ext))

    water_connect_array = np.array([])
    road_connect_array = np.array([])
    water_road_connect_array = np.array([])
    for im_f in tqdm(images):
        mask = np.array(Image.open(im_f))
        file_name = os.path.splitext(os.path.basename(im_f))[0]
        num_labels_wr = 0

        # 先统计含有水体的块数量分布
        if np.sum(mask == 3) >= 1:
            water = (mask == 3).astype(np.uint8)  # 01二值化
            num_labels_w, labels_w, stats_w, centers_w = cv2.connectedComponentsWithStats(water,
                                                                                  connectivity=8,
                                                                                  ltype=cv2.CV_16U)
            water_connect_array = np.append(water_connect_array, num_labels_w)
            num_labels_wr += num_labels_w

        # 再统计含有道路的块数量分布
        if np.sum(mask == 4) >= 1:

            road = (mask == 4).astype(np.uint8)  # 01二值化
            num_labels_r, labels_r, stats_r, centers_r = cv2.connectedComponentsWithStats(road,
                                                                                  connectivity=8,
                                                                                  ltype=cv2.CV_16U)
            road_connect_array = np.append(road_connect_array, num_labels_r)
            num_labels_wr += num_labels_r

        water_road_connect_array = np.append(water_road_connect_array, num_labels_wr)

    time.sleep(0.1)
    num, count = np.unique(water_road_connect_array, return_counts=True)
    print("image count:{}".format(water_road_connect_array.size))
    print(dict(zip(num, count)))

    num, count = np.unique(water_connect_array, return_counts=True)
    print("water count:{}".format(water_connect_array.size))
    print(dict(zip(num, count)))

    num, count = np.unique(road_connect_array, return_counts=True)
    print("road count:{}".format(road_connect_array.size))
    print(dict(zip(num, count)))


# _w: 水体(不含道路)且块数为1和2，暂不处理
# _r: 道路(不含水体)且块数为1和2，暂不处理
# _wr: 需要处理的
def pick_road_water(mask_path, ext):
    dirname = os.path.dirname(mask_path)
    basename = os.path.basename(mask_path)
    water_path = os.path.join(dirname, basename + "_w")
    if not os.path.exists(water_path):
        os.mkdir(water_path)
    road_path = os.path.join(dirname, basename + "_r")
    if not os.path.exists(road_path):
        os.mkdir(road_path)
    water_road_path = os.path.join(dirname, basename + "_wr")
    if not os.path.exists(water_road_path):
        os.mkdir(water_road_path)

    water_count = 0
    road_count = 0
    water_road_count = 0
    images = glob.glob(os.path.join(mask_path, '*.' + ext))
    for im_f in tqdm(images):
        file_name = os.path.splitext(os.path.basename(im_f))[0]
        mask = np.array(Image.open(im_f))

        # 其它类跳过
        if (3 not in mask) and (4 not in mask):
            continue

        # 把水体类型无需后处理的保存出来
        if np.sum(mask == 3) >= 1:
            water = (mask == 3).astype(np.uint8)  # 01二值化
            num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(water,
                                                                                  connectivity=8,
                                                                                  ltype=cv2.CV_16U)
            # 2:整块都是水体，或者只包含1块水体，3：包含2块水体
            # 由于存在水体数量是1或2块，但道路分块较多的情况，所以此处要加上不含道路
            if num_labels < 4 and (4 not in mask):
                shutil.copyfile(im_f, os.path.join(water_path, file_name + "." + ext))
                water_count += 1
                continue

        # 把道路类型无需后处理的保存出来
        if np.sum(mask == 4) >= 1:
            road = (mask == 4).astype(np.uint8)  # 01二值化
            num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(road,
                                                                                  connectivity=8,
                                                                                  ltype=cv2.CV_16U)
            # 参见水体过滤条件说明
            if num_labels < 4 and (3 not in mask):
                shutil.copyfile(im_f, os.path.join(road_path, file_name + "." + ext))
                road_count += 1
                continue

        # 剩下需要后处理的
        shutil.copyfile(im_f, os.path.join(water_road_path, file_name + "." + ext))
        water_road_count += 1

    time.sleep(0.1)
    print("water images:{}, road images:{}, process images:{}".format(water_count, road_count, water_road_count))


if __name__ == '__main__':

    # vis multi folder
    vis_images([r'/home/hou/Desktop/windowdata/temp/connected_component/img_testA',
                r'/home/hou/Desktop/windowdata/temp/connected_component/commit_result/3/5954_2020_12_14/results',
                r'/home/hou/Desktop/windowdata/temp/connected_component/commit_result/3/6480_2020_12_17/results',
                r'/home/hou/Desktop/windowdata/temp/connected_component/commit_result/3/hou_1/results',
                ],
               ['jpg', 'png', 'png', 'png'],
               ['image', '5830', '6480', 'hou'])


