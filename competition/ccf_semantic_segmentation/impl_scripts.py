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
import random
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes

from tqdm import tqdm
from PIL import Image as PILImage
from matplotlib import pyplot as plt


def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map

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
    plt.show()

def rm_small(input_dir,pro_dir,out_dir,thr=20,vis=True):
    """
    base proba image remove samall
    :param input_dir: input mask path
    :param pro_dir: proba image path
    :param out_dir: output path
    :param thr: pixel count thr
    :param vis: if true,vis
    :return:
    """
    pro_files = glob.glob(os.path.join(pro_dir, '*.npy'))
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(pro_files)
    color_map = get_color_map_list(256)
    for f in tqdm(pro_files):
        base_name=os.path.basename(f)
        base_name_noext=os.path.splitext(base_name)[0]
        mask_f=os.path.join(input_dir,base_name_noext+'.png')
        mask_array=np.array(PILImage.open(mask_f))

        all_pro = np.load(f)[0, ...]
        all_pro[:, :, 3]=0.0
        all_pro[:, :, 3][mask_array==3] = 0.99

        all_pro[:, :, 4]=0.0
        all_pro[:, :, 4][mask_array==4] = 1.0
        all_pro1 = all_pro.copy()

        res_map = np.squeeze(np.argmax(all_pro, axis=-1)).astype(np.uint8)
        water = (res_map == 3).astype(np.bool)
        remove_small_objects(water, thr, in_place=True)
        remove_small_holes(water, thr, in_place=True)
        all_pro[:, :, 3] = water.astype(np.int)

        road = (res_map == 4).astype(np.bool)
        remove_small_objects(road, thr, in_place=True)
        remove_small_holes(road, thr, in_place=True)
        all_pro[:, :, 4] = road.astype(np.int)
        vis_count = 0

        res_map = np.squeeze(np.argmax(all_pro, axis=-1)).astype(np.uint8)
        if vis and (np.sum(res_map==3)>10 or np.sum(res_map==4)>10):
            # if vis and image_line is not None :
            vis_count += 1
            vis_image_mask(str(vis_count) + '_' + base_name,
                           *[
                               {'image': PILImage.open(os.path.join('/home/data/yrj/train_tmp/ccf_lc/img_testA',
                                                                    base_name.replace('.npy', '.jpg')))},
                               {'pro': np.load(f)[0, :, :, 4]},
                               {'road': mask_array==4},
                               {'water': mask_array==3},

                               {'pro_max': np.argmax(np.load(f)[0, ...], axis=-1)},
                               {'res_map': res_map},
                           ],
                           )
        vis_fn = os.path.join(out_dir, base_name.replace('.npy', '.png'))

        pred_mask = PILImage.fromarray(res_map.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
        pred_mask.save(vis_fn)
    zip_file(out_dir,out_dir+'.zip')


def zip_file(dirname, zipfilename):
    import zipfile
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else:
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))
    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    for tar in tqdm(filelist):
        arcname = tar[len(os.path.dirname(dirname)):]
        # print arcname
        zf.write(tar, arcname)
    zf.close()

def pro2cls_image(pro_dir,out_cls_dir,vis=True):
    """
    pro img to commit cls png
    :param pro_dir:
    :param out_cls_dir:
    :param vis:
    :return:
    """
    pro_files = glob.glob(os.path.join(pro_dir, '*.npy'))
    os.makedirs(out_cls_dir, exist_ok=True)
    color_map = get_color_map_list(256)
    vis_count=0
    for f in tqdm(pro_files):
        base_name = os.path.basename(f)
        base_name_noext = os.path.splitext(base_name)[0]
        out_mask_f = os.path.join(out_cls_dir, base_name_noext + '.png')
        all_pro = np.load(f)[0, ...]
        res_map = np.squeeze(np.argmax(all_pro, axis=-1)).astype(np.uint8)

        if vis and (np.sum(res_map == 3) > 10 or np.sum(res_map == 4) > 10):
            # if vis and image_line is not None :
            vis_count += 1
            vis_image_mask(str(vis_count) + '_' + base_name,
                           *[
                               {'image': PILImage.open(os.path.join('/home/data/yrj/train_tmp/ccf_lc/img_testA',
                                                                    base_name.replace('.npy', '.jpg')))},
                               {'pro': np.load(f)[0, :, :, 4]},
                               {'road': res_map == 4},
                               {'water': res_map == 3},
                               {'res_map': res_map},
                           ],
                           )


        pred_mask = PILImage.fromarray(res_map.astype(np.uint8), mode='P')
        pred_mask.putpalette(color_map)
        pred_mask.save(out_mask_f)
    zip_file(out_cls_dir, out_cls_dir + '.zip')

if __name__ == '__main__':

    pro2cls_image(
        pro_dir='/home/data/yrj/train_tmp/ccf_lc/results_pro/s1m5+s1m4+s1m3+s1m2+s1m1+s1m6+s1m7+s1m8+s1m9+s1m10+s1m6d1+s1m8d1+s1m9d1/results',
        out_cls_dir='/tmp/out/results'
    )