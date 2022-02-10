import os
import sys
from multiprocessing import Pool

import cv2
import os
import sys
import glob
import time
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import OrderedDict
from sklearn.neighbors import KDTree
from rasterio.windows import Window


def read_image(path):
    """Read raster image from specified path (only 3 channles)"""
    with rasterio.open(path) as src:
        return src.read()[:3].transpose(1, 2, 0)


def read_borders(path,pixel_num,width,height):
    """Read 1-pixel borders from images (flattten to 1D array),
    retruns Dict[str, np.ndarray]
    """
    with rasterio.open(path) as src:
        # 10 pixel
        # left = src.read(window=Window(0, 0, 10, 600))[:3]
        # right = src.read(window=Window(590, 0, 10, 600))[:3]
        # top = src.read(window=Window(0, 0, 600, 10))[:3]
        # bottom = src.read(window=Window(0, 590, 600, 10))[:3]
        # 1 pixel
        # left = src.read(window=Window(0, 0, 1, 600))[:3]
        # right = src.read(window=Window(599, 0, 1, 600))[:3]
        # top = src.read(window=Window(0, 0, 600, 1))[:3]
        # bottom = src.read(window=Window(0, 599, 600, 1))[:3]
        # 1 pixel
        left = src.read(window=Window(0, 0, pixel_num, height))[:3]
        right = src.read(window=Window(width-1, 0, pixel_num, height))[:3]
        top = src.read(window=Window(0, 0, width, pixel_num))[:3]
        bottom = src.read(window=Window(0, height-1, width, pixel_num))[:3]
        # 10 pixel
        # left = src.read(window=Window(0, 0, 10, 399))[:3]
        # right = src.read(window=Window(389, 0, 10, 399))[:3]
        # top = src.read(window=Window(0, 0, 399, 10))[:3]
        # bottom = src.read(window=Window(0, 389, 399, 10))[:3]
    return dict(
        left=left.flatten(),
        right=right.flatten(),
        top=top.flatten(),
        bottom=bottom.flatten(),
    )


def read_descriptors(paths,pixel_num,width,height):
    """Read image descriptors (borders)"""
    descriptors = OrderedDict()

    with tqdm(paths) as p_paths:
        for path in p_paths:
            id = os.path.basename(path)
            descriptor = read_borders(path,pixel_num,width,height)
            descriptors[id] = descriptor
    return descriptors


def get_neighbours(x1, x2, threshold=0.98):
    """Extract two nearest neigbours for each side of image,
    if [distance to first]/[distance to second] < threshold assume that we find matching image"""
    x2_kd = KDTree(x2)
    distances, keys = x2_kd.query(x1, k=2)
    distances_rel = (distances[:, 0] + 0.0001) / (distances[:, 1] + 0.0001)
    keys = keys[:, 0]
    keys[distances_rel > threshold] = -1
    return keys


def check_validity(x1, x2):
    """Check neighbours validity, check that left neighbour for image right image is the same
    as right nighbour for left image"""
    x1 = x1.copy()
    x2 = x2.copy()

    for i, k1 in enumerate(x1):
        if k1 != -1:
            j = x2[k1]
            if j == -1:
                x1[i] = -1

    for i, k1 in enumerate(x2):
        if k1 != -1:
            j = x1[k1]
            if j == -1:
                x2[i] = -1

    return x1, x2


def make_cluster(cluster, nb, k, visited, pos):
    """Create clusters of images accoring to its nearest neighbours"""
    cluster[k] = pos
    visited.add(k)
    node = nb[k]
    for _pos, connection in node.items():
        if connection not in visited and connection != -1:
            x, y = pos
            if _pos == 'left':
                x = x - 1
            if _pos == 'right':
                x = x + 1
            if _pos == 'top':
                y = y - 1
            if _pos == 'bottom':
                y = y + 1
            make_cluster(cluster, nb, connection, visited, pos=(x, y))


def normalize_cluster(cluster):
    """Looking for min X and min Y position values in cluster
    and shift them to make cluster position started from (0, 0)"""
    xs = [c[0] for c in cluster.values()]
    ys = [c[1] for c in cluster.values()]

    min_x = min(xs)
    min_y = min(ys)

    cluster_ = {}
    for k, v in cluster.items():
        cluster_[k] = (v[0] - min_x, v[1] - min_y)
    return cluster_


def get_xy_max(cluster):
    """Looking for max X nd Y positions in cluster"""
    xs = [c[0] for c in cluster.values()]
    ys = [c[1] for c in cluster.values()]
    return max(xs), max(ys)


def reverse_cluster(cluster):
    """Reverse clusters keys and values (encode name: poisiton -> position: name)"""
    return {v: k for k, v in cluster.items()}





stitcher = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
def images_stitcher(rev_cluster,paths,x,y_min,y_max,output_data):

    imgs = []
    output_data_name=''
    for y in range(y_min, y_max + 1):
        try:
            k = rev_cluster[(x, y)]
            img = cv2.imread(paths[k])
            output_data_name=os.path.basename(paths[k])
            input_data_imgs_name_list = output_data_name.split('_')
            output_data_name = str(input_data_imgs_name_list[0]) + '_' + str(input_data_imgs_name_list[1]) + '_' + str(
                input_data_imgs_name_list[2]) + '_' + str(input_data_imgs_name_list[3])
            if img is None:
                print("can't read image " + paths[k])
                sys.exit(-1)
            imgs.append(img)
        except KeyError:
            pass

    try:
        (status, pano) = stitcher.stitch(imgs)
        if status != cv2.STITCHER_OK:
            print("不能拼接图片"+str(output_data_name))
        print("拼接成功："+str(output_data_name))
        cv2.imwrite(os.path.join(output_data,output_data_name+'.jpg'),pano)
    except Exception as e:
        print("图片拼接错误："+str(output_data_name),e)


if __name__ == '__main__':
    input_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_05_08_form_casia\Images'
    output_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\temp_101'
    width=399
    height=399
    pixel_num=1
    if not os.path.exists(output_data):
        os.makedirs(output_data)
        # reading descriptors
    paths = glob.glob(os.path.join(input_data,'*.jpg'))

    descriptors = read_descriptors(paths,pixel_num,width,height)
    # extract descriptors for each side of image
    left_arr = np.array([descriptors[k]["left"] for k in descriptors.keys()])[:, ::8]
    right_arr = np.array([descriptors[k]["right"] for k in descriptors.keys()])[:, ::8]
    top_arr = np.array([descriptors[k]["top"] for k in descriptors.keys()])[:, ::8]
    bottom_arr = np.array([descriptors[k]["bottom"] for k in descriptors.keys()])[:, ::8]
    # extarcting nearest negbours for each image according to its border descriptor
    # heavy step, may take about 10 minutes
    lr_keys = get_neighbours(left_arr, right_arr)
    rl_keys = get_neighbours(right_arr, left_arr)
    tb_keys = get_neighbours(top_arr, bottom_arr)
    bt_keys = get_neighbours(bottom_arr, top_arr)
    # check consistency of neighbours
    lr_keys_, rl_keys_ = check_validity(lr_keys, rl_keys)
    tb_keys_, bt_keys_ = check_validity(tb_keys, bt_keys)
    # create neighbours dict
    neighbours = {}
    for i, k in enumerate(descriptors.keys()):
        neighbours[i] = dict(
            left=lr_keys_[i],
            right=rl_keys_[i],
            top=tb_keys_[i],
            bottom=bt_keys_[i],
        )

    # create clusters of groupped images recursively going through neighbours
    visited = set()
    clusters = []

    for i in range(len(paths)):
        if i not in visited:
            cluster = {}
            make_cluster(cluster, neighbours, i, visited, (0, 0))
            clusters.append(cluster)

    # Visualization of mosaic (clusters)
    for cluster in clusters:
        # if len(cluster.keys()) < 40:  # skip small clusters
        #     continue
        cluster = normalize_cluster(cluster)
        rev_cluster = reverse_cluster(cluster)

        x_min, y_min = 0, 0
        x_max, y_max = get_xy_max(cluster)

        for x in range(x_min, x_max + 1):
            images_stitcher(rev_cluster,paths,x,y_min,y_max,output_data)











