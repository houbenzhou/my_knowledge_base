# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:18:13 2021

@author: 13354
"""

import cv2, time, rasterio, os
import numpy as np
from matplotlib import pyplot as plt
from rasterio import Affine
from rasterio.enums import Resampling
import multiprocessing
from rasterio.windows import Window
from multiprocessing import Pool, Manager, Lock

"""
resize 50分之一后，大概1秒一张
对两个文件夹内的基准影像和测试影像遍历进行粗匹配，使用多进程加速
"""
# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


def resample_raster(raster, imgs, outdir, scale=0.05):
    t = raster.transform
    # raw_shape = (raster.height, raster.width)
    # rescale the metadata
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
    height = int(raster.height * scale)
    width = int(raster.width * scale)
    height1 = raster.height
    width1 = raster.width

    profile = raster.profile
    profile.update(transform=transform, driver='GTiff', height=height, width=width)

    data = raster.read(  # Note changed order of indexes, arrays are band, row, col order not row, col, band
        out_shape=(raster.count, height1, width1),
        # resampling=Resampling.bilinear,
    )

    data = np.transpose(data, axes=(1, 2, 0))
    print(data.shape)
    data = cv2.resize(data, (width, height))
    cv2.imwrite((outdir + '/' + imgs[:-4] + '-%s-%s-' % (raster.height, raster.width) + '.tif'), data)
    return data


def det_comp(img_path, imgs, outdir, scale=50):
    "得到给定路径图片的关键点和描述子"

    with  rasterio.open(img_path) as ds_b:
        raw_shape = (ds_b.height, ds_b.width)
        base_img = resample_raster(ds_b, imgs, outdir, scale=1 / scale)
        base_img_r = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = sift.detectAndCompute(base_img_r, None)
    return keypoints, descriptors, base_img_r


def multiprocess_match(test_img_dir, base_img_dir, base_imgs, out_txt_dir, out_plot_dir, name_t):
    st_coordinate = time.time()
    scale = 50  # 原图缩放为原来的1/50
    # st = time.time()
    # for name_t in test_imgs:

    test_img_path = os.path.join(test_img_dir, name_t)
    test_img_r = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), 0)  # todo 有的图用rasterio打开没有cv快，但大部分都比cv快
    # raw_shape_t = test_img.shape
    raw_shape_s = name_t.split('-')
    raw_shape_t = (int(raw_shape_s[-3]), int(raw_shape_s[-2]))
    print(raw_shape_t)

    # test_img_r = cv2.resize(test_img, (test_img.shape[0] // scale, test_img.shape[1] // scale))
    # test_img_r = cv2.cvtColor(test_img_r, cv2.COLOR_BGR2GRAY)

    keypoints_t, descriptors_t = sift.detectAndCompute(test_img_r, None)
    # keypoints_t, descriptors_t, raw_shape_t, test_img_r=det_comp(test_img_path,scale)

    min_points_dist = np.inf
    matched_base = None
    print(len(base_imgs))

    for name_b in base_imgs:
        base_img_path = os.path.join(base_img_dir, name_b)

        # keypoints_b, descriptors_b, raw_shape_b, _ = det_comp(base_img_path, scale)

        base_img_r = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8), 0)
        keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        if matches[0].distance == 0:  # 如果出现关键点距离为0 ，很大概率匹配正确
            matched_base = name_b
            break

        points_dist = 0
        # 如果没有出现距离为0的关键点，那只能遍历完所有图片才能得到匹配结果
        for p in matches[:10]:  # 计算距离最短的10个关键点的距离
            points_dist += p.distance

        if min_points_dist > points_dist:
            min_points_dist = points_dist
            matched_base = name_b

    if len(base_imgs) > 0:
        base_imgs.remove(matched_base)
    # else:
    #     break

    # 再匹配一次，获取对应的匹配结果
    # keypoints_b, descriptors_b, raw_shape_b, base_img_r = det_comp(os.path.join(base_img_dir, matched_base), scale)

    base_img_r = cv2.imdecode(np.fromfile(os.path.join(base_img_dir, matched_base), dtype=np.uint8), 0)
    raw_shape = matched_base.split('-')
    raw_shape_b = (int(raw_shape[-3]), int(raw_shape[-2]))
    print(raw_shape_b)
    keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

    # feature matching
    matches = bf.match(descriptors_b, descriptors_t)
    matches = sorted(matches, key=lambda x: x.distance)

    # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配,
    # 粗匹配由于缩放，会使得两个匹配点间的distance变大

    img3 = cv2.drawMatches(base_img_r, keypoints_b, test_img_r, keypoints_t, matches[:20], test_img_r,
                           flags=2)

    # 得到距离最小的匹配点
    nearest_point = matches[0]
    qdx = nearest_point.queryIdx
    tdx = nearest_point.trainIdx
    point_b = keypoints_b[qdx].pt  # (x,y),  up left as raw point
    point_t = keypoints_t[tdx].pt

    # 以此最近点为中心，选取测试影像里的部分区域进行精匹配
    tile_size_t = 1200
    # 关键点在原图上大概的像素坐标
    rt_x = raw_shape_t[1] * point_t[0] / test_img_r.shape[1]
    rt_y = raw_shape_t[0] * point_t[1] / test_img_r.shape[0]
    xmin_t = max(int(rt_x - tile_size_t / 2), 0)
    ymin_t = max(int(rt_y - tile_size_t / 2), 0)
    xmax_t = min(xmin_t + tile_size_t, raw_shape_t[1])
    ymax_t = min(ymin_t + tile_size_t, raw_shape_t[0])

    rt_rect = [xmin_t, ymin_t, xmax_t, ymax_t]  # 以关键点为中心取边长为tile_size的矩形

    # 基础影像也可以选择对应的中心点进行精匹配
    tile_size_b = tile_size_t
    rb_x = raw_shape_b[1] * point_b[0] / base_img_r.shape[1]
    rb_y = raw_shape_b[0] * point_b[1] / base_img_r.shape[0]
    xmin_b = max(int(rb_x - tile_size_b / 2), 0)
    ymin_b = max(int(rb_y - tile_size_b / 2), 0)
    xmax_b = min(xmin_b + tile_size_b, raw_shape_b[1])
    ymax_b = min(ymin_b + tile_size_b, raw_shape_b[0])
    rb_rect = [xmin_b, ymin_b, xmax_b, ymax_b]

    # txt_name = "%s2%s.txt" % (matched_base.split('.')[0], name_t.split('.')[0])
    txt_name = "%s-%s.txt" % (matched_base.split('-')[0], name_t.split('-')[0])

    # png_name = "%s2%s.png" % (matched_base.split('.')[0], name_t.split('.')[0])
    png_name = "%s-%s.png" % (matched_base.split('-')[0], name_t.split('-')[0])

    with open(os.path.join(out_txt_dir, txt_name), 'w', encoding='utf-8') as txt:  # 写入txt文件
        txt.write("%d %d %d %d \n" % (rt_rect[0], rt_rect[1], rt_rect[2], rt_rect[3]))  # 写入测试图片的
        txt.write("%d %d %d %d \n" % (rb_rect[0], rb_rect[1], rb_rect[2], rb_rect[3]))

    plt.imshow(img3)
    # plt.show()
    save_path = os.path.join(out_plot_dir, png_name)
    plt.savefig(save_path)
    end_coordinate = time.time()
    print("coordinate time cost is  {}".format(end_coordinate - st_coordinate))


def trans_coordinate(_rela, txt_dir, base_img_dir, test_img_dir, save_dir, transform_dir, lefttop_coordinate_dir):
    st_s = time.time()

    name_b, name_t = _rela.split('.')[0].split('-')
    base_img_path = os.path.join(base_img_dir, name_b + '.tif')
    test_img_path = os.path.join(test_img_dir, name_t + '.tif')

    # 读取txt文件中的匹配区域
    txt_file = os.path.join(txt_dir, _rela.split('.')[0] + '.txt')
    with open(txt_file, 'r') as txt:
        rec_t = txt.readline().strip('\n')
        rec_t = rec_t.split(' ')[:4]

        rec_b = txt.readline().strip('\n')
        rec_b = rec_b.split(' ')[:4]

        rec_t = list(map(lambda x: int(x), rec_t))
        rec_b = list(map(lambda x: int(x), rec_b))

    with rasterio.open(base_img_path) as ds_b:
        _block_b = ds_b.read(window=Window(rec_b[0], rec_b[1], rec_b[2] - rec_b[0], rec_b[3] - rec_b[1]))

        base_block = _block_b[:3, :, :]
        base_block = np.ma.transpose(base_block, [1, 2, 0])
        base_block = cv2.cvtColor(base_block, cv2.COLOR_BGR2GRAY)

        keypoints_b, descriptors_b = sift.detectAndCompute(base_block, None)

        base_transform = ds_b.transform
        base_height = ds_b.height
        base_width = ds_b.width

        with rasterio.open(test_img_path) as ds_t:
            # 起始列行，取的宽高
            _block = ds_t.read(window=Window(rec_t[0], rec_t[1], rec_t[2] - rec_t[0], rec_t[3] - rec_t[1]))
            test_block = _block[:3, :, :]
            test_block = np.ma.transpose(test_block, [1, 2, 0])

            _height = ds_t.height
            _wid = ds_t.width

        test_block = cv2.cvtColor(test_block, cv2.COLOR_BGR2GRAY)

        keypoints_t, descriptors_t = sift.detectAndCompute(test_block, None)

        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配
        img3 = cv2.drawMatches(base_block, keypoints_b, test_block, keypoints_t, matches[:20], test_block,
                               flags=2)

        plt.imshow(img3)
        # plt.show()
        save_path = os.path.join(save_dir, "%s-%s.png" % (name_b, name_t))
        plt.savefig(save_path)

        # cv2.imshow('img3', img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite((save_path),img3)

        nearest_point = matches[0]
        qdx = nearest_point.queryIdx
        tdx = nearest_point.trainIdx
        point_b = list(keypoints_b[qdx].pt)  # (x,y),  up left as raw point
        point_t = list(keypoints_t[tdx].pt)

        # 这对距离最近的匹配点各自在原图上的位置
        point_b[0] += rec_b[0]  # 匹配点在基础影像上的像素坐标
        point_b[1] += rec_b[1]

        point_t[0] += rec_t[0]  # 匹配点在测试影像上的像素坐标
        point_t[1] += rec_t[1]

        # 得到测试影像的左上角原点在基础影像上的像素坐标
        test_raw_point_x = point_b[0] - point_t[0]
        test_raw_point_y = point_b[1] - point_t[1]

        # 测试影像右下角在基础影像上的像素坐标
        _right = test_raw_point_x + _wid
        _bot = test_raw_point_y + _height

        # 构建像素坐标转经纬度坐标的转换矩阵
        transform_metric = np.array(base_transform)
        print(transform_metric)
        transform_metric = np.reshape(transform_metric, (3, 3))
        np.savetxt(os.path.join(transform_dir, name_b + '-' + name_t + '.txt'), transform_metric, fmt='%.50f')  # 使用默认分割符（空格），保留四位小数
        # 测试影像左上角和右下角的经纬度坐标转换结果
        point_arr1 = np.array([_right, _bot, 1])
        point_arr2 = np.array([test_raw_point_x, test_raw_point_y, 1])

        geo_right, geo_bot, _ = np.matmul(transform_metric, point_arr1)
        geo_left, geo_top, _ = np.matmul(transform_metric, point_arr2)
        with open(os.path.join(lefttop_coordinate_dir, name_b + '-' + name_t + '.txt'), 'w',
                  encoding='utf-8') as txt_c:  # 左上角坐标和baseimage_name写入txt文件
            txt_c.write('%s %.1f %.1f\n' % (name_b, test_raw_point_x, test_raw_point_y))
        print("geo_right %f,geo_left %f,geo_bot %f,geo_top %f" % (geo_right, geo_left, geo_bot, geo_top))

        ee_s = time.time()
        print('single time is:', ee_s - st_s)
        # TODO 是先把转换矩阵存入txt还是等bbox算出来之后再运行脚本得出目标的地理坐标，可以先存好，但是得把小数点后的位数精确好


if __name__ == "__main__":
    input_test_image = r"E:\new_iob\image-matching\testimage"  # 测试影像文件夹
    input_base_image = r"E:\new_iob\image-matching\baseimage"  # 基准影像文件夹
    out_match_dir = r"E:\new_iob\image-matching\out_dir"  # 粗匹配输出文件夹
    out_lefttop_coordinate_dir = r"E:\new_iob\image-matching\coordinate"  # 小图左上角位置在大图上的坐标
    out_transform_dir = r"E:\new_iob\image-matching\transform"  # 像素坐标到地理坐标的映射关系

    test_imgs_raw = os.listdir(input_test_image)
    test_imgs_raw = list(filter(lambda x: x.endswith('tif'), test_imgs_raw))
    base_imgs_raw = os.listdir(input_base_image)
    base_imgs_raw = list(filter(lambda x: x.endswith('tif'), base_imgs_raw))

    out_txt_dir = os.path.join(out_match_dir, 'txt')
    out_plot_dir = os.path.join(out_match_dir, 'match_plot')
    os.makedirs(out_txt_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)

    base_resize_img_dir = os.path.join(input_base_image, 'resize')
    test_resize_img_dir = os.path.join(input_test_image, 'resize')
    os.makedirs(base_resize_img_dir, exist_ok=True)
    os.makedirs(test_resize_img_dir, exist_ok=True)

    st = time.time()

    # for imgs in base_imgs_raw:
    #     det_comp(os.path.join(base_img_dir, imgs),imgs,base_resize_img_dir)

    print('begin to baseimage resize...')
    print('Parent process %s.' % os.getpid())
    p = Pool(3)  # 控制进程数，根据计算机硬件决定（cpu和内存）
    for i in range(len(base_imgs_raw)):
        p.apply_async(det_comp,
                      args=(os.path.join(input_base_image, base_imgs_raw[i]), base_imgs_raw[i], base_resize_img_dir,))
    print('Waiting for all resize subprocesses done...')
    p.close()
    p.join()
    print('All resize subprocesses done.')

    print('begin to testimage resize...')
    print('Parent process %s.' % os.getpid())
    p = Pool(6)  # 控制进程数，根据计算机硬件决定（cpu和内存）
    for i in range(len(test_imgs_raw)):
        p.apply_async(det_comp,
                      args=(os.path.join(input_test_image, test_imgs_raw[i]), test_imgs_raw[i], test_resize_img_dir,))
    print('Waiting for all resize subprocesses done...')
    p.close()
    p.join()
    print('All resize subprocesses done.')

    base_imgs = os.listdir(base_resize_img_dir)
    base_imgs = list(filter(lambda x: x.endswith('tif'), base_imgs))
    test_imgs = os.listdir(test_resize_img_dir)
    test_imgs = list(filter(lambda x: x.endswith('tif'), test_imgs))
    print(base_imgs)
    print(test_imgs)
    print('begin to match...')
    for imgs in test_imgs:
        multiprocess_match(test_resize_img_dir, base_resize_img_dir, base_imgs, out_txt_dir, out_plot_dir, imgs)

    save_picture_dir = out_match_dir
    os.makedirs(out_lefttop_coordinate_dir, exist_ok=True)
    os.makedirs(out_transform_dir, exist_ok=True)

    rela_imgs = os.listdir(out_plot_dir)
    rela_imgs = list(filter(lambda x: x.endswith('png'), rela_imgs))

    # for _rela in rela_imgs: # 单进程精匹配
    #     trans_coordinate(_rela, out_txt_dir, input_base_image, input_test_image, save_picture_dir, out_transform_dir,
    #                      out_lefttop_coordinate_dir)

    print('begin to fine match...') # 多进程精匹配
    print('Parent process %s.' % os.getpid())
    p = Pool(3)  # 控制进程数，根据计算机硬件决定（cpu和内存）
    for _rela in rela_imgs:
        p.apply_async(trans_coordinate,
                      args=(_rela, out_txt_dir, input_base_image, input_test_image, save_picture_dir, out_transform_dir,
                         out_lefttop_coordinate_dir,))
    print('Waiting for all fine match subprocesses done...')
    p.close()
    p.join()
    print('All fine match subprocesses done.')

    end = time.time()
    print("total time cost is  {}".format(end - st))

    # with Manager() as MG:  #多进程匹配
    #     base_imgs_g = MG.list(base_imgs) #将base_imgs作为多进程间的全局变量
    #     print(len(base_imgs_g))
    #     print('Parent process %s.' % os.getpid())
    #     p = Pool(2)  #控制进程数，根据计算机硬件决定（cpu和内存）
    #     for i in range(len(test_imgs)): #len(test_imgs)
    #     #for imgs in test_imgs:
    #         j = str(i%4)
    #         print(j)
    #         #print(os.path.join(base_resize_img_dir, j))
    #         p.apply_async(multiprocess_match, args=(test_resize_img_dir,os.path.join(base_resize_img_dir, j),base_imgs_g,out_txt_dir, out_plot_dir, test_imgs[i],))
    #         #p.apply_async(multiprocess_match, args=(test_resize_img_dir,base_resize_img_dir,base_imgs_g, out_txt_dir, out_plot_dir, test_imgs[i],))
    #     print('Waiting for all subprocesses done...')
    #     p.close()
    #     p.join()
    #     #multiprocess_match(test_img_dir,base_resize_img_dir,base_imgs_g, out_txt_dir, out_plot_dir, test_imgs[len(test_imgs)-1])
    #     print('All subprocesses done.')
    #     print(base_imgs_g) #检验全局变量是否更新

