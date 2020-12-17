# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import os
import shutil

import numpy as np
import rasterio


def copy_allfiles(src, dest, file_name_path=None):
    """
    copy所有文件到另一个文件夹

    | 如果file_name_path是None，src为文件夹，则将src文件夹拷贝到dest文件夹
    | 如果file_name_path是None，src为文件，则将src文件拷贝到dest文件夹
    | 如果file_name_path不是None，src必须为文件夹，从file_name_path中获取src文件夹中的文件名，并将相应的文件拷贝到dest文件夹

    :param src: 原文件夹
    :type src: str
    :param dest: 目标文件夹
    :type dest: str
    :param file_name_path: 获取文件的file_name
    :type file_name_path: None or str
    """
    if file_name_path is None:
        if os.path.isdir(src):
            src_files = os.listdir(src)
            for file_name in src_files:
                full_file_name = os.path.join(src, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dest)
        else:
            shutil.copy(src, dest)
    else:
        src_files = os.listdir(file_name_path)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)


def stretch_n(bands, lower_percent=1, higher_percent=99, tile_h=15000, tile_w=15000):
    """
    16 to 8
    :param bands:
    :param lower_percent:
    :param higher_percent:
    :return:
    """
    is_transpose = False
    if np.argmin(bands.shape) == 0:
        is_transpose = True
    if is_transpose:
        bands = np.transpose(bands, (1, 2, 0))
    out = np.zeros_like(bands, dtype=np.uint8)
    h, w, n = bands.shape
    # tile_h,tile_w=15000,15000
    index_h, index_w = h // tile_h + 1, w // tile_w + 1
    total_task = index_h * index_w * n
    task_index = 0
    # print('processing {}/{}'.format(task_index, total_task))
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        for j in range(index_h):
            for k in range(index_w):
                tmp_h, tmp_w = min((j + 1) * tile_h, h), min((k + 1) * tile_w, w)
                t = a + (bands[j * tile_h:tmp_h, k * tile_w:tmp_w, i] - c) * (b - a) / (d - c)
                t[t < a] = a
                t[t > b] = b
                out[j * tile_h:tmp_h, k * tile_w:tmp_w, i] = t.astype(np.uint8)
                task_index += 1
                # print('processing {}/{}'.format(task_index, total_task))

        # t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        # t[t < a] = a
        # t[t > b] = b
        # out[:, :, i] = t.astype(np.uint8)
    if is_transpose:
        out = np.transpose(out, (2, 0, 1))
    return out


# 16位图像转8位图像
def u16_to_u8(in_image, out_image):
    in_f = rasterio.open(in_image)
    prof = in_f.profile
    prof.update({'dtype': rasterio.uint8})
    with rasterio.open(out_image, 'w', **prof) as wf:
        wf.write(stretch_n(in_f.read()))
