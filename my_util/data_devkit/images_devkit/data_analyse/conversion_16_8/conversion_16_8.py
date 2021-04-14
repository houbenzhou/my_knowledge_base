# !/usr/bin/env python3
# coding=utf-8


import numpy as np
import rasterio


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
    print('processing {}/{}'.format(task_index, total_task))
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
                print('processing {}/{}'.format(task_index, total_task))

        # t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        # t[t < a] = a
        # t[t > b] = b
        # out[:, :, i] = t.astype(np.uint8)
    if is_transpose:
        out = np.transpose(out, (2, 0, 1))
    return out


def u16_to_u8(in_image, out_image):
    in_f = rasterio.open(in_image)
    prof = in_f.profile
    prof.update({'dtype': rasterio.uint8})
    with rasterio.open(out_image, 'w', **prof) as wf:
        wf.write(stretch_n(in_f.read()))


if __name__ == '__main__':
    # u16_to_u8(in_image='/home/hou/Desktop/windowdata/temp/科目1-2/科目1-2/03发布数据-光学-全色.tiff',
    #           out_image='/home/data/temp.tif')
    u16_to_u8(in_image='/home/data/windowdata/temp/科目2-2/科目2-2/02发布数据-sar.tiff',
              out_image='/home/data/temp_sar.tif')
