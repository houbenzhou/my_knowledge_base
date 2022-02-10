# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:18:13 2021

@author: 13354
"""
import codecs

import cv2, time, rasterio, os
import numpy as np
import os

from PIL import Image
from matplotlib import pyplot as plt
from rasterio import Affine
from rasterio.enums import Resampling
import multiprocessing
from rasterio.windows import Window
from multiprocessing import Pool, Manager, Lock
import xml.etree.ElementTree as ET

# sift
sift = cv2.SIFT_create()
# Brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


def multiprocess_match(test_img_dir, base_img_dir, base_imgs, out_all_txt_dir, out_txt_dir, out_plot_dir):
    """
        计算一组图像数据之间的配准关系；输出这组数据中聚类到一起的txt文件，每一组具体配准的文件，可视化每一组具体配准的文件。
        :param test_img_dir: 输入查询图像路径
        :type test_img_dir: str
        :param base_img_dir:  待查询图像路径
        :type base_img_dir: str
        :param base_imgs: 待查询图像数据名
        :type base_imgs: list
        :param out_all_txt_dir: 文件以查询图像名命名，保存查询图像名以及待查询图像名拼接合成名，以及左上点相对坐标
        :type out_all_txt_dir: str
        :param out_txt_dir:  文件查询图像名以及待查询图像名拼接合成名命名，保存查询图像名以及待查询图像名拼接合成名，以及左上点相对坐标
        :type out_txt_dir: str
        :param out_plot_dir: 文件查询图像名以及待查询图像名拼接合成名命名，查询文件与被查询文件关键点映射的可视化图
        :type out_plot_dir: str

    """
    name_t = base_imgs[0]
    base_imgs.remove(name_t)

    st_coordinate = time.time()

    test_img_path = os.path.join(test_img_dir, name_t)
    test_img_r = cv2.imdecode(np.fromfile(test_img_path, dtype=np.uint8), 0)  # todo 有的图用rasterio打开没有cv快，但大部分都比cv快

    keypoints_t, descriptors_t = sift.detectAndCompute(test_img_r, None)
    print(len(base_imgs))
    matched_base_list = []
    for name_b in base_imgs:
        base_img_path = os.path.join(base_img_dir, name_b)

        base_img_r = cv2.imdecode(np.fromfile(base_img_path, dtype=np.uint8), 0)
        keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        points_dist = 0
        # 如果没有出现距离为0的关键点，那只能遍历完所有图片才能得到匹配结果
        for p in matches[:10]:  # 计算距离最短的10个关键点的距离
            points_dist += p.distance

        if points_dist <= 1000:
            print(",,,,,,,,,,,,"+str(name_b)+str(points_dist))
            matched_base = name_b
            matched_base_list.append(matched_base)

    txt_all_name = "%s.txt" % (name_t.split('.')[0])
    all_txt = open(os.path.join(out_all_txt_dir, txt_all_name), 'w', encoding='utf-8')
    if matched_base_list == []:
        all_txt.close()
    matched_base_temp_list=[]
    for matched_base_temp in matched_base_list:
        base_img_r = cv2.imdecode(np.fromfile(os.path.join(base_img_dir, matched_base_temp), dtype=np.uint8), 0)
        keypoints_b, descriptors_b = sift.detectAndCompute(base_img_r, None)

        # feature matching
        matches = bf.match(descriptors_b, descriptors_t)
        matches = sorted(matches, key=lambda x: x.distance)

        # queryIdx代表的特征点序列是keypoints_b中的，trainIdx代表的特征点序列是keypoints_t中的，此时这两张图中的特征点相互匹配

        img3 = cv2.drawMatches(base_img_r, keypoints_b, test_img_r, keypoints_t, matches[:20], test_img_r,
                               flags=2)

        # 得到距离最小的匹配点
        nearest_point = matches[0]
        qdx = nearest_point.queryIdx
        tdx = nearest_point.trainIdx
        point_b = keypoints_b[qdx].pt  # (x,y),  up left as raw point
        point_t = keypoints_t[tdx].pt

        # 得到测试影像的左上原点在基础影像上的像素坐标
        test_raw_point_x = point_b[0] - point_t[0]
        test_raw_point_y = point_b[1] - point_t[1]
        # # 得到测试影像的中心原点在基础影像上的像素坐标
        # test_raw_point_x = point_b[0] - point_t[0] + 200
        # test_raw_point_y = point_b[1] - point_t[1] + 200

        txt_name = "%s-%s.txt" % (name_t.split('.')[0], matched_base_temp.split('.')[0])
        png_name = "%s-%s.png" % (name_t.split('.')[0], matched_base_temp.split('.')[0])
        matched_base_temp_list.append(matched_base_temp.split('.')[0])
        with open(os.path.join(out_txt_dir, txt_name), 'w', encoding='utf-8') as txt:  # 写入txt文件
            txt.write('%s-%s %.1f %.1f\n' % (
            name_t.split('.')[0], matched_base_temp.split('.')[0], test_raw_point_x, test_raw_point_y))
        all_txt.write('%s-%s %.1f %.1f\n' % (
            name_t.split('.')[0], matched_base_temp.split('.')[0], test_raw_point_x, test_raw_point_y))
        plt.imshow(img3)
        save_path = os.path.join(out_plot_dir, png_name)
        plt.savefig(save_path)
        end_coordinate = time.time()
        print("coordinate time cost is  {}".format(end_coordinate - st_coordinate))
    all_txt.close()
    for matched_base_name in matched_base_temp_list:
        matched_base_name_jpg = matched_base_name+".jpg"
        if matched_base_name_jpg in base_imgs:
            base_imgs.remove(matched_base_name_jpg)
    if base_imgs ==[]:
        pass
    else:
        multiprocess_match(test_img_dir, base_img_dir, base_imgs, out_all_txt_dir, out_txt_dir, out_plot_dir)


def splicing_small_pic(input_image_data, input_label_data, first_name,second_name, x_left, y_top, out_voc_data):
    """
        对输入的小图进行匹配，并将合成的大图输出到out_voc文件夹中
        :param input_image_data: 图像路径
        :type input_image_data: str
        :param input_label_data: 标签路径
        :type input_label_data: str
        :param first_name: 用于查询的图片文件名，输出保存的大图文件也以此命名
        :type first_name: list
        :param second_name: 查询的图片文件名，用来与first合并的文件名
        :type second_name: str
        :param x_left:  第一幅图左上点在第二幅图的位置x方向
        :type x_left: int
        :param y_top: 第一幅图左上点在第二幅图的位置y方向
        :type y_top: int
        :param out_voc_data: 输出的voc数据集路径
        :type out_voc_data: str

    """
    out_label_path = os.path.join(out_voc_data, "Annotations")
    out_image_path = os.path.join(out_voc_data, "Images")
    out_coordinate_path = os.path.join(out_voc_data, "coordinate")
    os.makedirs(out_label_path, exist_ok=True)
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_coordinate_path, exist_ok=True)

    first_image_name=first_name+'.jpg'
    second_image_name=second_name+'.jpg'
    first_label_name=first_name+'.xml'
    second_label_name=second_name+'.xml'

    first_image_file=os.path.join(input_image_data,first_image_name)
    second_image_file = os.path.join(input_image_data, second_image_name)
    first_label_file = os.path.join(input_label_data, first_label_name)
    second_label_file = os.path.join(input_label_data, second_label_name)
    out_image_file = os.path.join(out_image_path, first_label_name)
    out_label_file = os.path.join(out_label_path, first_label_name)

    # 获取用于查询的小图在输出大图的左上点相对位置
    x_coordinate = 0
    y_coordinate = 0
    first_coordinate_name=first_name+'.txt'
    out_origin_first_image_on_coordinate_path = os.path.join(out_coordinate_path, first_coordinate_name)
    if os.path.exists(out_origin_first_image_on_coordinate_path):
        file = open(out_origin_first_image_on_coordinate_path, "r", encoding="utf-8", errors="ignore")
        try:
            while True:
                mystr = file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                list_ = mystr.split(' ')
                x_coordinate = int(float(list_[0].strip()))
                y_coordinate = int(float(list_[1].strip()))
        except IOError:
            print(IOError)
        file.close()

    # 如果输出匹配大图的文件夹影像不存在，则从输入文件夹获取相应的影像
    if os.path.exists(out_image_file):
        pass
    else:
        out_image_file=first_image_file

    if os.path.exists(out_label_file):
        pass
    else:
        out_label_file=first_label_file


    x_=int(abs(float(x_left)))
    y_=int(abs(float(y_top)))
    first_origin_img_x = 0
    first_origin_img_y = 0
    first_img_x=0
    first_img_y=0
    second_img_x=0
    second_img_y=0
    if (~int(float(x_left)))>0:
        first_img_x = 0
        first_origin_img_x = x_coordinate
        second_img_x = x_+ x_coordinate
    if ((~int(float(x_left)))<=0)&(x_coordinate>=x_):
        first_img_x = 0
        first_origin_img_x = x_coordinate
        second_img_x = x_coordinate-x_
    if ((~int(float(x_left)))<=0)&(x_coordinate<x_):
        first_img_x = x_-x_coordinate
        first_origin_img_x = x_
        second_img_x = 0

    if (~int(float(y_top)))>0:
        first_img_y = 0
        first_origin_img_y = y_coordinate
        second_img_y = y_+y_coordinate
    if (~int(float(y_top)<=0))&(y_coordinate>=y_):
        first_img_y = 0
        first_origin_img_y=y_coordinate
        second_img_y = y_coordinate-y_
    if (~int(float(y_top))<=0)&(y_coordinate<y_):
        first_img_y = y_-y_coordinate
        first_origin_img_y=y_
        second_img_y = 0

    # 获取用于匹配的图像标签
    first_label_anno_file = ET.parse(out_label_file)
    first_label_bbox_list=[]
    second_label_bbox_list=[]

    for obj in first_label_anno_file.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        first_label_bbox_list.append({"bbox":bbox,"label":cls})

    second_label_anno_file = ET.parse(second_label_file)
    for obj in second_label_anno_file.findall("object"):
        cls = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [int(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        second_label_bbox_list.append({"bbox":bbox,"label":cls})

    # 计算匹配出来的图矩形框是否存在完全重合的影像来验证图像匹配成功
    match = False
    for first_label_bbox in first_label_bbox_list:
        for second_label_bbox in second_label_bbox_list:
            first_label_xmin = first_label_bbox['bbox'][0]
            first_label_ymin = first_label_bbox['bbox'][1]
            first_label_xmax = first_label_bbox['bbox'][2]
            first_label_ymax = first_label_bbox['bbox'][3]

            second_label_xmin = second_label_bbox['bbox'][0]
            second_label_ymin = second_label_bbox['bbox'][1]
            second_label_xmax = second_label_bbox['bbox'][2]
            second_label_ymax = second_label_bbox['bbox'][3]

            first_label_xmin = first_label_xmin + first_origin_img_x
            first_label_ymin = first_label_ymin + first_origin_img_y
            first_label_xmax = first_label_xmax + first_origin_img_x
            first_label_ymax = first_label_ymax + first_origin_img_y

            second_label_xmin = second_label_xmin + second_img_x
            second_label_ymin = second_label_ymin + second_img_y
            second_label_xmax = second_label_xmax + second_img_x
            second_label_ymax = second_label_ymax + second_img_y

            first_areas = (first_label_xmax - first_label_xmin) * (first_label_ymax - first_label_ymin )
            second_areas = (second_label_xmax - second_label_xmin ) * (second_label_ymax - second_label_ymin )

            inter=0
            # 计算相交面积，不重叠的时候为0
            if (max(first_label_xmin, second_label_xmin) <= min(first_label_xmax, second_label_xmax))&(max(first_label_ymin, second_label_ymin) <= min(first_label_ymax, second_label_ymax)):
                inter = (min(first_label_xmax, second_label_xmax)-max(first_label_xmin, second_label_xmin))*(min(first_label_ymax, second_label_ymax)-max(first_label_ymin, second_label_ymin))

            # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
            ovr = inter / (first_areas + second_areas - inter)
            # 保留IoU小于阈值的box
            if ovr >=1.0:
                match = True
    if match:

        first_img = Image.open(out_image_file)
        second_img = Image.open(second_image_file)

        # 获取大图的宽和高
        W = 800
        H = 800
        if (~int(float(x_left))) > 0:
            W = max(first_img.size[0] , second_img.size[0] + second_img_x)
        if ((~int(float(x_left)))<=0)&(x_coordinate>=x_):
            W = first_img.size[0]
        if ((~int(float(x_left))) <= 0) & (x_coordinate < x_):
            W = first_img.size[0] + first_img_x

        if (~int(float(y_top))) > 0:
            H = max(first_img.size[1], second_img.size[1] + second_img_y)
        if (~int(float(y_top) <= 0)) & (y_coordinate >= y_):
            H = first_img.size[1]
        if (~int(float(y_top)) <= 0) & (y_coordinate < y_):
            H = first_img.size[1] + first_img_y


        # 新建大图，并使用小图填充
        img_big = Image.new('RGB', (int(W), int(H)))
        img_big.paste(first_img, (first_img_x, first_img_y))
        img_big.paste(second_img, (second_img_x,second_img_y))
        img_big.save(os.path.join(out_image_path, first_image_name))

        # 保存用于匹配的小图在大图上的左上点坐标
        coordinate_txt = open(out_origin_first_image_on_coordinate_path, 'w', encoding='utf-8')
        coordinate_txt.write('%.1f %.1f\n' % (first_origin_img_x,first_origin_img_y))
        coordinate_txt.close()

        # 保存新的大图坐标
        out_label_file = os.path.join(out_label_path, first_label_name)

        out_label_bbox_list_first = []
        out_label_bbox_list_second = []
        for first_label_bbox in first_label_bbox_list:
            first_label_xmin = first_label_bbox['bbox'][0]
            first_label_ymin = first_label_bbox['bbox'][1]
            first_label_xmax = first_label_bbox['bbox'][2]
            first_label_ymax = first_label_bbox['bbox'][3]


            first_label_xmin = first_label_xmin + first_origin_img_x
            first_label_ymin = first_label_ymin + first_origin_img_y
            first_label_xmax = first_label_xmax + first_origin_img_x
            first_label_ymax = first_label_ymax + first_origin_img_y
            # out_label_bbox_list_first.append([first_label_xmin,first_label_ymin,first_label_xmax,first_label_ymax])
            out_label_bbox_list_first.append(
                {"bbox": [first_label_xmin,first_label_ymin,first_label_xmax,first_label_ymax],
                 "label": first_label_bbox['label']})

        for second_label_bbox in second_label_bbox_list:

            second_label_xmin = second_label_bbox['bbox'][0]
            second_label_ymin = second_label_bbox['bbox'][1]
            second_label_xmax = second_label_bbox['bbox'][2]
            second_label_ymax = second_label_bbox['bbox'][3]
            second_label_xmin = second_label_xmin + second_img_x
            second_label_ymin = second_label_ymin + second_img_y
            second_label_xmax = second_label_xmax + second_img_x
            second_label_ymax = second_label_ymax + second_img_y
            out_label_bbox_list_second.append({"bbox":[second_label_xmin, second_label_ymin, second_label_xmax, second_label_ymax],"label":second_label_bbox['label']})


        for i in out_label_bbox_list_first:
            for j in out_label_bbox_list_second:
                first_areas = (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])
                second_areas = (j["bbox"][2] - j["bbox"][0]) * (j["bbox"][3] - j["bbox"][1])

                inter = 0
                # 计算相交面积，不重叠的时候为0
                if (max(i["bbox"][0], j["bbox"][0]) <= min(i["bbox"][2], j["bbox"][2])) & (
                        max(i["bbox"][1], j["bbox"][1]) <= min(i["bbox"][3], j["bbox"][3])):
                    inter = (min(i["bbox"][2], j["bbox"][2]) - max(i["bbox"][0], j["bbox"][0])) * (min(i["bbox"][3], j["bbox"][3]) - max(i["bbox"][1], j["bbox"][1]))

                # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
                ovr = inter / (first_areas + second_areas - inter)
                # 保留IoU小于阈值的box
                if ovr >= 0.5:
                    out_label_bbox_list_second.remove(j)
        for j in out_label_bbox_list_second:
            out_label_bbox_list_first.append(j)

        with codecs.open(out_label_file, "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC' + '</folder>\n')

            xml.write('\t<filename>' + first_image_name + '</filename>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>' + str(W) + '</width>\n')
            xml.write('\t\t<height>' + str(H) + '</height>\n')
            xml.write('\t\t<depth>' + str(3) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')

            for i in out_label_bbox_list_first:
                xmin=i["bbox"][0]
                ymin=i["bbox"][1]
                xmax=i["bbox"][2]
                ymax=i["bbox"][3]
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + i["label"] + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(round(xmin))) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(round(ymin))) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(round(xmax))) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(round(ymax))) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')

            xml.write('</annotation>')







def splicing_small_images_into_large_images(input_image_data, input_label_data, all_txt_dir, out_voc_data):
    """
        对输入的小图进行匹配，并将合成的大图输出到out_voc文件夹中
        :param input_image_data: 图像路径
        :type input_image_data: str
        :param input_label_data: 标签路径
        :type input_label_data: str
        :param all_txt_dir: 用于查询的第一张图像的文件命名，计算所有与它相交的数据；此信息是sift匹配后计算的结果
        :type all_txt_dir: str
        :param out_voc_data: 输出的voc数据集路径
        :type out_voc_data: str

    """
    all_txt_list = os.listdir(all_txt_dir)
    for all_txt_name in all_txt_list:
        all_txt=os.path.join(all_txt_dir,all_txt_name)
        file = open(all_txt, "r", encoding="utf-8", errors="ignore")
        try:
            while True:
                mystr = file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                list_ = mystr.split(' ')
                pic_name=list_[0].strip()
                first_name=pic_name.split("-")[0]
                second_name=pic_name.split("-")[1]
                x_left=list_[1].strip()
                y_top=list_[2].strip()

                splicing_small_pic(input_image_data, input_label_data,first_name,second_name,x_left,y_top,out_voc_data)
        except IOError:
            print(IOError)


if __name__ == "__main__":
    input_data = r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_05_08_form_casia"
    output_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\temp\get_01_05_08_form_casia'

    input_image_data = os.path.join(input_data,"Images")
    input_label_data=os.path.join(input_data,"Annotations")

    out_all_txt_dir = os.path.join(output_data, 'all_txt')
    out_txt_dir = os.path.join(output_data, 'txt')
    out_plot_dir = os.path.join(output_data, 'match_plot')
    out_voc_data = os.path.join(output_data, "voc")
    os.makedirs(out_txt_dir, exist_ok=True)
    os.makedirs(out_plot_dir, exist_ok=True)
    os.makedirs(out_all_txt_dir, exist_ok=True)
    os.makedirs(out_voc_data, exist_ok=True)
    imgs_clusters = {}
    input_data_imgs_names = os.listdir(input_image_data)
    # for input_data_imgs_name in input_data_imgs_names:
    #     input_data_imgs_name_list=input_data_imgs_name.split('_')
    #     key_name=str(input_data_imgs_name_list[0])+'_'+str(input_data_imgs_name_list[1])+'_'+str(input_data_imgs_name_list[2])+'_'+str(input_data_imgs_name_list[3])
    #     if imgs_clusters.get(key_name)==None:
    #         imgs_clusters[key_name]=[input_data_imgs_name]
    #     else:
    #         imgs_clusters[key_name].append(input_data_imgs_name)
    # for imgs_cluster in imgs_clusters:
    #     imgs_cluster_list=imgs_clusters[imgs_cluster]
    #     multiprocess_match(input_image_data,input_image_data, imgs_cluster_list, out_all_txt_dir,out_txt_dir, out_plot_dir)

    all_txt_dir=out_all_txt_dir
    os.makedirs(out_voc_data, exist_ok=True)
    splicing_small_images_into_large_images(input_image_data,input_label_data,all_txt_dir,out_voc_data)