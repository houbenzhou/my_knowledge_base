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

import numpy as np
from PIL import Image


def _rgb(v):
    """
    获取RGB颜色
   :param v: 十六进制颜色码
   :return: RGB颜色值
       """
    r, g, b = v[1:3], v[3:5], v[5:7]
    return int(r, 16), int(g, 16), int(b, 16)


# 获取颜色表列表
def _get_codes_list():
    # 颜色表
    # dark = _rgb("#404040")
    # gray = _rgb("#eeeeee")
    # light = _rgb("#f8f8f8")
    # white = _rgb("#ffffff")
    # cyan = _rgb("#3bb2d0")
    # blue = _rgb("#3887be")
    # bluedark = _rgb("#223b53")
    # denim = _rgb("#50667f")
    # navy = _rgb("#28353d")
    # navydark = _rgb("#222b30")
    # purple = _rgb("#8a8acb")
    # teal = _rgb("#41afa5")
    # green = _rgb("#56b881")
    # yellow = _rgb("#f1f075")
    # mustard = _rgb("#fbb03b")
    # orange = _rgb("#f9886c")
    # red = _rgb("#e55e5e")
    # pink = _rgb("#ed6498")
    color_codes_list = []
    color_codes_list.append(_rgb("#f9886c"))
    color_codes_list.append(_rgb("#ed6498"))
    color_codes_list.append(_rgb("#eeeeee"))
    color_codes_list.append(_rgb("#f8f8f8"))
    color_codes_list.append(_rgb("#ffffff"))
    color_codes_list.append(_rgb("#3bb2d0"))
    color_codes_list.append(_rgb("#3887be"))
    color_codes_list.append(_rgb("#f1f075"))
    color_codes_list.append(_rgb("#fbb03b"))
    color_codes_list.append(_rgb("#404040"))
    color_codes_list.append(_rgb("#e55e5e"))
    color_codes_list.append(_rgb("#223b53"))
    color_codes_list.append(_rgb("#50667f"))
    color_codes_list.append(_rgb("#28353d"))
    color_codes_list.append(_rgb("#222b30"))
    color_codes_list.append(_rgb("#8a8acb"))
    color_codes_list.append(_rgb("#41afa5"))
    color_codes_list.append(_rgb("#56b881"))

    return color_codes_list


def save_pattle_png(image, color_codes, out_file):
    if out_file.endswith('png'):
        r = sorted(color_codes.items(), key=lambda d: d[1])
        palette = [color_value for class_color in r for color_value in class_color[0]]
        out = Image.fromarray(np.squeeze(image), mode="P")
        out.putpalette(palette)
        out.save(out_file, optimize=True)
    else:
        raise Exception('out_file should end with png')


# 输入为概率图，输出为mask

def probability_plot_to_result_png(input_data, road_grass_th, water_th, output_data):
    input_ids = os.listdir(input_data)
    if not os.path.exists(output_data):
        os.makedirs(output_data)
    for i in input_ids:
        logit = np.load(os.path.join(input_data, i))
        if road_grass_th != 1.0:
            if road_grass_th > 0.5:
                logit[:, :, :, 4][logit[:, :, :, 4] < road_grass_th] = 0.0
                # logit[:,:,:,5][logit[:, :, :, 5] < road_grass_th] = 0.0
            else:
                logit[:, :, :, 4][logit[:, :, :, 4] > road_grass_th] = 1.0
        if water_th != 1.0:
            if water_th > 0.5:
                logit[:, :, :, 4][logit[:, :, :, 4] < water_th] = 0.0
                # logit[:,:,:,5][logit[:, :, :, 5] < road_grass_th] = 0.0
            else:
                logit[:, :, :, 4][logit[:, :, :, 4] > water_th] = 1.0
        res_map = np.squeeze(np.argmax(logit[0, :, :, :], axis=-1)).astype(np.uint8)

        im = Image.fromarray(np.uint8(res_map))
        color_codes_segobject = {}

        color_continuous_codes_list = _get_codes_list()
        for color_codes_id in range(7):
            color_codes_segobject[color_continuous_codes_list[color_codes_id]] = color_codes_id

        save_pattle_png(im, color_codes_segobject, os.path.join(output_data, i.split(".")[0] + ".png"))


if __name__ == '__main__':
    input_data = '/home/data/windowdata/temp/connected_component/gailvtu/results'
    output_data = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp/out_gailvtu'
    road_grass_th = 0.5
    water_th = 0.5
    probability_plot_to_result_png(input_data, road_grass_th, water_th, output_data)

# input_data = '/home/hou/Desktop/windowdata/temp/connected_component/commit_result/0.595459_r_origin+ske_close8_dilate1_center_r-only_dis6.20b2rmsm/A149304.png'
# output_data = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp/out_gailvtu'
# # input_ids = os.listdir(input_data)
# if not os.path.exists(output_data):
#     os.makedirs(output_data)
# im = Image.open(input_data)  # im是Image对象
# img = np.asarray(im)  #
# # # img = np.asarray(im)
# # np.save('test.npy', img)
# test = Image.fromarray(img, im.mode)
# im.save("./test1.png")
# test.save("./test.png")
# # from scipy import misc
# #
# # misc.imsave("./test.png", img)
# input_data = '/home/data/windowdata/temp/connected_component/gailvtu/A153112.npy'
# output_data = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp/out_gailvtu'
# if not os.path.exists(output_data):
#     os.makedirs(output_data)
# logit = np.load(input_data)
#
# res_map = np.squeeze(np.argmax(logit[0, :, :, :], axis=-1)).astype(np.uint8)
#
# im = Image.fromarray(np.uint8(res_map))
# im.save("./test.png")
