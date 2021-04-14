#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  :
@Email   :
@Time    : 14:36
@Site    :
@File    : CompressImage.py
@Software: PyCharm
"""

"""
将16位遥感图像压缩至8位，并保持色彩一致
"""

import gdal
import os
import glob
import numpy as np


def read_tiff(input_file):
    """
    读取影像
    :param input_file:输入影像
    :return:波段数据，仿射变换参数，投影信息、行数、列数、波段数
    """

    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    couts = dataset.RasterCount

    array_data = np.zeros((couts, rows, cols))

    for i in range(couts):
        band = dataset.GetRasterBand(i + 1)
        array_data[i, :, :] = band.ReadAsArray()

    return array_data, geo, proj, rows, cols, 3


def compress(origin_16, output_8):
    array_data, geo, proj, rows, cols, couts = read_tiff(origin_16)

    compress_data = np.zeros((couts, rows, cols))

    for i in range(couts):
        band_max = np.max(array_data[i, :, :])
        band_min = np.min(array_data[i, :, :])

        cutmin, cutmax = cumulativehistogram(array_data[i, :, :], rows, cols, band_min, band_max)

        compress_scale = (cutmax - cutmin) / 255

        for j in range(rows):
            for k in range(cols):
                if (array_data[i, j, k] < cutmin):
                    array_data[i, j, k] = cutmin

                if (array_data[i, j, k] > cutmax):
                    array_data[i, j, k] = cutmax

                compress_data[i, j, k] = (array_data[i, j, k] - cutmin) / compress_scale

    write_tiff(output_8, compress_data, rows, cols, couts, geo, proj)


def write_tiff(output_file, array_data, rows, cols, counts, geo, proj):
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file, cols, rows, counts, gdal.GDT_Byte)

    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    for i in range(counts):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array_data[i, :, :])


def cumulativehistogram(array_data, rows, cols, band_min, band_max):
    """
    累计直方图统计
    """

    # 逐波段统计最值

    gray_level = int(band_max - band_min + 1)
    gray_array = np.zeros(gray_level)

    counts = 0
    for row in range(rows):
        for col in range(cols):
            gray_array[int(array_data[row, col] - band_min)] += 1
            counts += 1

    count_percent2 = counts * 0.02
    count_percent98 = counts * 0.98

    cutmax = 0
    cutmin = 0

    for i in range(1, gray_level):
        gray_array[i] += gray_array[i - 1]
        if (gray_array[i] >= count_percent2 and gray_array[i - 1] <= count_percent2):
            cutmin = i + band_min

        if (gray_array[i] >= count_percent98 and gray_array[i - 1] <= count_percent98):
            cutmax = i + band_min

    return cutmin, cutmax


if __name__ == '__main__':
    origin_16 = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\16\demo_16.tiff'
    output_8 = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\8_percent\demo_8.tiff'

    # origin_16 = r"D:\ZY3_MUX_E133.3_N47.7_20160722_L1A0003484148\ZY3_MUX_E133.3_N47.7_20160722_L1A0003484148.tiff"
    # output_8 = r"D:\new22.tif"
    compress(origin_16, output_8)
