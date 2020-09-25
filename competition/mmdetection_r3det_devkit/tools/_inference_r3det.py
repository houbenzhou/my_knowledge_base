#!/usr/bin/env python
# encoding: utf-8
import logging
import os
import sys
import time

import cv2
import numpy as np
import rasterio
from mmdet.apis import inference_detector, init_detector
from rasterio.plot import reshape_as_image
from rasterio.windows import Window

"""
影像数据目标检测
"""


class R3detEstimation(object):
    def __init__(self, model_path, cfg, classes, tile_size=512, tile_offset=256):
        self.model_path = model_path
        self.cfg = cfg
        self.classes = classes
        self.tile_size = tile_size
        self.tile_offset = tile_offset

    def estimation_img(self, input_data, category_name, out_data, out_name, nms_thresh=0.3,
                       score_thresh=0.5):
        """
        进行影像数据目标检测
        """

        result, _ = self._estimation_img(input_data, category_name, out_data,
                                         out_name, nms_thresh,
                                         score_thresh)

        return result

    def estimation_dir(self, input_data, category_name, out_data_path, nms_thresh=0.3,
                       score_thresh=0.5):
        """
       input_data为目录,进行影像数据目标检测
       """
        image_datas_list = os.listdir(input_data)
        i = 0
        _start_time = time.time()
        for image_name in image_datas_list:
            images_pth = os.path.join(input_data, image_name)
            out_name = image_name.split('.')[0]
            # out_data = os.path.join(out_data_path, out_name) + ".txt"
            if self._is_image_file(images_pth):
                i = i + 1
                start_time = time.time()
                _, num_objects = self._estimation_img(images_pth, category_name, out_data_path, out_name, nms_thresh,
                                                      score_thresh)

                print('{}:detected {} targets in {:.2f}s'.format(images_pth, num_objects, time.time() - start_time))
                logging.info(
                    '{}:detected {} targets in {:.2f}s'.format(images_pth, num_objects, time.time() - start_time))
            else:
                # print('{}： is not an images'.format(images_pth))
                logging.warning('{}： is not an images'.format(images_pth))

    def _estimation_img(self, input_data, category_name, out_data, out_name, nms_thresh=0.3,
                        score_thresh=0.5):
        """
        进行影像数据目标检测
        """
        self.input_data = input_data
        self.category_name = category_name
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

        with rasterio.open(self.input_data) as ds:
            width_block = ds.width // self.tile_offset + 1
            height_block = ds.height // self.tile_offset + 1

            # all_boxes = []
            all_boxes = {cls: '' for cls in self.classes}
            if (ds.height <= self.tile_size) | (ds.width <= self.tile_size):
                all_boxes = self._get_bbox(ds, -1, -1,
                                           all_boxes)
                logging.info('()： Width and height less than or equal to ()'.format(self.input_data, self.tile_size))
            else:
                # 记录程序运行进度条
                p = 0
                for i in range(height_block):
                    for j in range(width_block):
                        all_boxes = self._get_bbox(ds, j, i,
                                                   all_boxes)
                        p += 1
                        self._view_bar(p, (height_block) * (width_block))
                print('')
                logging.info('()： Width and height is greater than ()'.format(self.input_data, self.tile_size))

            mult_categorys_nms = {cls: '' for cls in self.classes}
            for classe_name in self.classes:
                single_category_list = []
                for temp_bbox in all_boxes[classe_name].split('\n'):
                    temp_bbox = temp_bbox.split()
                    if temp_bbox != []:
                        rs = map(float, temp_bbox)
                        temp_bbox = list(rs)
                        single_category_list.append(temp_bbox)
                single_category_list = np.array(single_category_list)
                if (single_category_list != np.array([])):
                    keep = self.rnms(single_category_list, nms_thresh)
                    single_category_list = single_category_list[keep, :]
                    bboxes = self.rdets2points(single_category_list)
                    for i in range(bboxes.shape[0]):
                        resstr = '{:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                        ps = list(bboxes[i][:-1])
                        score = float(bboxes[i][-1])
                        resstr = resstr.format(score, *ps)
                        mult_categorys_nms[classe_name] += resstr
            # 对all_boxes中所有的框整体去重
            num_objects = 0

            # num_objects = self.write_origin_file(out_data, out_name, mult_categorys_nms, num_objects)
            num_objects = self.write_gaofen4_file(out_data, out_name, mult_categorys_nms, num_objects)
        return 1, num_objects

    def _is_image_file(self, input_data):
        """
        输入数据是否为影像文件
        通过后缀名判断
        """
        try:
            with rasterio.open(input_data) as ds:
                data_is_image = True
        except Exception as e:
            data_is_image = False

        return data_is_image

    def _get_bbox(self, ds, j, i,
                  all_boxes):

        height = ds.height
        width = ds.width
        block_xmin = j * self.tile_offset
        block_ymin = i * self.tile_offset
        if (j == -1) & (i == -1):
            block_xmin = 0
            block_ymin = 0
            block = np.zeros([3, ds.height, ds.width], dtype=np.uint8)
            img = ds.read(window=Window(block_xmin, block_ymin, ds.width, ds.height))

        else:
            block_xmax = block_xmin + self.tile_size
            block_ymax = block_ymin + self.tile_size
            if height <= block_ymax:
                block_ymin = height - self.tile_size
            if width <= block_xmax:
                block_xmin = width - self.tile_size
            block = np.zeros([3, self.tile_size, self.tile_size], dtype=np.uint8)
            img = ds.read(window=Window(block_xmin, block_ymin, self.tile_size, self.tile_size))

        block[:, :img.shape[1], :img.shape[2]] = img[:3, :, :]
        block = reshape_as_image(block)
        block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)

        # 执行预测
        model = init_detector(self.cfg, self.model_path, device='cuda:0')
        result = inference_detector(model, block)
        result2str = self._det2str(result, block_xmin,block_ymin,self.classes)

        for classe_name in self.classes:
            for result_transform in result2str[classe_name].split('\n'):
                if result_transform == '':
                    result_transform = ''
                else:
                    result_transform = result_transform + ('\n')
                all_boxes[classe_name] += result_transform
        return all_boxes

    def rdets2points(self, rbboxes):
        """Convert detection results to a list of numpy arrays.

        Args:
            rbboxes (np.ndarray): shape (n, 6), xywhap encoded

        Returns:
            rbboxes (np.ndarray): shape (n, 9), x1y1x2y2x3y3x4y4p
        """
        x = rbboxes[:, 0]
        y = rbboxes[:, 1]
        w = rbboxes[:, 2]
        h = rbboxes[:, 3]
        a = rbboxes[:, 4]
        prob = rbboxes[:, 5]
        cosa = np.cos(a)
        sina = np.sin(a)
        wx, wy = w / 2 * cosa, w / 2 * sina
        hx, hy = -h / 2 * sina, h / 2 * cosa
        p1x, p1y = x - wx - hx, y - wy - hy
        p2x, p2y = x + wx - hx, y + wy - hy
        p3x, p3y = x + wx + hx, y + wy + hy
        p4x, p4y = x - wx + hx, y - wy + hy
        return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob], axis=-1)

    def _det2str(self, result,block_xmin,block_ymin, classes):
        mcls_results = {cls: '' for cls in classes}
        for label in range(len(result)):
            bboxes = result[label]
            cls_name = classes[label]
            for i in range(bboxes.shape[0]):
                resstr = '{:.6f} {:.6f} {:.6f} {:.6f} {:.12f} {:.12f}\n'
                # ps = list(bboxes[i][:-1])
                # score = float(bboxes[i][-1])
                bboxes[i][0]=bboxes[i][0]+block_xmin
                bboxes[i][1] = bboxes[i][1] + block_ymin
                resstr = resstr.format(*bboxes[i])
                mcls_results[cls_name] += resstr
        return mcls_results

    def rnms(self, boxes, iou_threshold):

        keep = []
        boxes[:, -1]
        order = boxes[:, -1].argsort()[::-1]
        num = boxes.shape[0]

        suppressed = np.zeros((num), dtype=np.int)

        for _i in range(num):
            # if len(keep) >= max_output_size:
            #     break

            i = order[_i]
            if suppressed[i] == 1:
                continue
            keep.append(i)
            r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
            area_r1 = boxes[i, 2] * boxes[i, 3]
            for _j in range(_i + 1, num):
                j = order[_j]
                if suppressed[i] == 1:
                    continue
                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

                except:
                    """
                      cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                      error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                    """
                    # print(r1)
                    # print(r2)
                    inter = 0.9999

                if inter >= iou_threshold:
                    suppressed[j] = 1

        return np.array(keep, np.int64)

    def _view_bar(self, num, total):
        """
        进度条
        """
        rate = float(num) / float(total)
        rate_num = int(rate * 100)
        r = '\r[%s%s]%d%%,%d' % (">" * rate_num, "-" * (100 - rate_num), rate_num, num)
        sys.stdout.write(r)
        sys.stdout.flush()

    def write_origin_file(self, out_data, out_name, mult_categorys_nms, num_objects):
        out_data = os.path.join(out_data, out_name + '.txt')

        with open(out_data, 'a') as file_out:
            for classe_name in self.classes:
                for temp_bbox in mult_categorys_nms[classe_name].split('\n'):
                    if temp_bbox.split() != []:
                        num_objects = num_objects + 1
                        outline = classe_name + ' ' + temp_bbox
                        file_out.write(outline + '\n')

    def write_gaofen4_file(self, out_data, out_name, mult_categorys_nms, num_objects):
        from lxml.etree import Element, SubElement, tostring
        out_data = os.path.join(out_data, out_name + '.xml')

        node_root = Element('annotation')

        # SOURCE
        node_source = SubElement(node_root, 'source')
        node_filename = SubElement(node_source, 'filename')
        node_filename.text = '4.tif'
        node_origin = SubElement(node_source, 'origin')
        node_origin.text = 'GF2/GF3'

        # RESEARCH
        node_research = SubElement(node_root, 'research')
        node_version = SubElement(node_research, 'version')
        node_version.text = '4.0'
        node_provider = SubElement(node_research, 'provider')
        node_provider.text = 'Company/School of team'
        node_author = SubElement(node_research, 'author')
        node_author.text = 'team name'
        node_pluginname = SubElement(node_research, 'pluginname')
        node_pluginname.text = 'Airplane Detection and Recognition'
        node_pluginclass = SubElement(node_research, 'pluginclass')
        node_pluginclass.text = 'Detection'
        node_time = SubElement(node_research, 'time')
        node_time.text = '2020-07-2020-11'

        # OBJECTS
        node_objects = SubElement(node_root, 'objects')

        for label in mult_categorys_nms:
            cls_name = label
            bboxes = []
            for temp_bbox in mult_categorys_nms[label].split('\n'):
                temp_bbox = temp_bbox.split()
                if temp_bbox != []:
                    rs = map(float, temp_bbox)
                    temp_bbox = list(rs)
                    bboxes.append(temp_bbox)
            for i in range(len(bboxes)):
                node_object = SubElement(node_objects, 'object')

                node_coordinate = SubElement(node_object, 'coordinate')
                node_coordinate.text = 'pixel'
                node_type = SubElement(node_object, 'type')
                node_type.text = 'rectangle'
                node_description = SubElement(node_object, 'description')
                node_description.text = 'None'
                # POSSIBLERESULT
                node_possibleresult = SubElement(node_object, 'possibleresult')
                node_name = SubElement(node_possibleresult, 'name')
                node_name.text = cls_name
                node_probability = SubElement(node_possibleresult, 'probability')
                node_probability.text = str(bboxes[i][1])
                # POINT
                node_points = SubElement(node_object, 'points')
                node_point1 = SubElement(node_points, 'point')
                node_point1.text = str(int(bboxes[i][1])) + ',' + str(int(bboxes[i][2]))
                node_point2 = SubElement(node_points, 'point')
                node_point2.text = str(int(bboxes[i][3])) + ',' + str(int(bboxes[i][4]))
                node_point3 = SubElement(node_points, 'point')
                node_point3.text = str(int(bboxes[i][5])) + ',' + str(int(bboxes[i][6]))
                node_point4 = SubElement(node_points, 'point')
                node_point4.text = str(int(bboxes[i][7])) + ',' + str(int(bboxes[i][8]))
                node_point5 = SubElement(node_points, 'point')
                node_point5.text = str(int(bboxes[i][1])) + ',' + str(int(bboxes[i][2]))

        xml_ = tostring(node_root, pretty_print=True, encoding='UTF-8')
        with open(out_data, 'wb') as file_out:
            file_out.write(xml_)
