# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:38:35 2021

@author: 13354
"""
import os
import numpy as np
import time


# detect_coordinate = list(filter(lambda x: x.endswith('.txt'), detect_coordinate_dir))
def dect_coordinate_trans(out_file, transform_dir, lefttop_coordinate_dir, detect_coordinate_dir, detect_coordinate):
    transform_list = os.listdir(transform_dir)
    for test_txt in detect_coordinate:
        txt_file = os.path.join(detect_coordinate_dir, test_txt)
        list_read = open(txt_file).readlines()
        for line in list_read:
            line = line.strip()
            print(line)
            line = line.split(',')  # [:4]
            test_img_name = line[0]
            class_encode = line[9]
            coordinate = line[1:9]
            coordinate = list(map(lambda x: float(x), coordinate))
            print(coordinate)
            for transform_name in transform_list:
                base_name, test_name = transform_name.split('.')[0].split('-')
                if test_name == str(test_img_name):
                    transform_metric = np.loadtxt(os.path.join(transform_dir, transform_name[:-4] + '.txt'))
                    lefttop_coordinate = os.path.join(lefttop_coordinate_dir, transform_name[:-4] + '.txt')
            with open(lefttop_coordinate, 'r') as txt:
                lefttop_cod = txt.readline().strip('\n')
                lefttop_cod = lefttop_cod.split(' ')
                base_img_name = lefttop_cod[0]
                print(base_img_name)
                lefttop_coordinate = [float(lefttop_cod[1]), float(lefttop_cod[2])]
                print(lefttop_coordinate)

            # 得到检测框在基础影像上的像素坐标
            left_mid_x = lefttop_coordinate[0] + (coordinate[0] + coordinate[2] + coordinate[4] + coordinate[6]) / 4
            left_mid_y = lefttop_coordinate[1] + (coordinate[1] + coordinate[3] + coordinate[5] + coordinate[7]) / 4
            print(left_mid_x)
            print(left_mid_y)

            point_mid = np.array([left_mid_x, left_mid_y, 1])
            geo_mid_x, geo_mid_y, _ = np.matmul(transform_metric, point_mid)

            print("geo_mid_x %f,geo_mid_y %f" % (geo_mid_x, geo_mid_y))

            out_file.write("%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s,%.15f,%.15f \n"
                           % (test_img_name, base_img_name, coordinate[0], coordinate[1], coordinate[2], coordinate[3]
                              , coordinate[4], coordinate[5], coordinate[6], coordinate[7], class_encode, geo_mid_x,
                              geo_mid_y))


if __name__ == "__main__":
    outfile = r"E:\new_iob\image-matching\out.txt"  # 提交结果修改部分
    input_transform_dir = r"E:\new_iob\image-matching\transform"  # 像素的映射关系
    input_lefttop_coordinate_dir = r"E:\new_iob\image-matching\coordinate"  # 小图左上坐标到地理坐标角位置在大图上的坐标
    input_detect_coordinate_dir = r"E:\new_iob\image-matching\detect"  # 测试预测结果
    input_detect_coordinate = os.listdir(input_detect_coordinate_dir)  # 测试预测结果list
    print(input_detect_coordinate)

    st = time.time()
    out_file = open(outfile, "w")
    dect_coordinate_trans(out_file, input_transform_dir, input_lefttop_coordinate_dir, input_detect_coordinate_dir,
                          input_detect_coordinate)
    out_file.close()
    end = time.time()
    print("cost time is {}".format(end - st))