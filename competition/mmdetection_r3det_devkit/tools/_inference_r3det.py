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
    def __init__(self, model_path, cfg, classes, tile_size=4096, tile_offset=2048):
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

        result, _ = self._estimation_img(input_data, category_name, os.path.join(out_data, out_name) + ".txt",
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
            out_data = os.path.join(out_data_path, out_name) + ".txt"
            if self._is_image_file(images_pth):
                i = i + 1
                start_time = time.time()
                _, num_objects = self._estimation_img(images_pth, category_name, out_data, out_name, nms_thresh,
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
            num_objects = self.write_origin_file(out_data, mult_categorys_nms, num_objects)

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
        result2str = self._det2str(result, self.classes)

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

    def _det2points(self, result, classes):
        mcls_results = {cls: '' for cls in classes}
        for label in range(len(result)):
            bboxes = self.rdets2points(result[label])
            cls_name = classes[label]
            for i in range(bboxes.shape[0]):
                resstr = '{:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'
                ps = list(bboxes[i][:-1])
                score = float(bboxes[i][-1])
                resstr = resstr.format(score, *ps)
                mcls_results[cls_name] += resstr
        return mcls_results

    def _det2str(self, result, classes):
        mcls_results = {cls: '' for cls in classes}
        for label in range(len(result)):
            bboxes = result[label]
            cls_name = classes[label]
            for i in range(bboxes.shape[0]):
                resstr = '{:.6f} {:.6f} {:.6f} {:.6f} {:.12f} {:.12f}\n'
                # ps = list(bboxes[i][:-1])
                # score = float(bboxes[i][-1])
                resstr = resstr.format(*bboxes[i])
                mcls_results[cls_name] += resstr
        return mcls_results

    def _get_blobs(self, im):
        """将影像转换为网络输入"""
        blobs = {}
        blobs['data'], im_scale_factors = self._get_image_blob(im)

        return blobs, im_scale_factors

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in (600,):
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > 1000:
                im_scale = float(1000) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def _im_list_to_blob(self, ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def _clip_boxes(self, boxes, im_shape):
        """Clip boxes to image boundaries."""
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
        return boxes

    def _bbox_transform_inv(self, boxes, deltas):
        """
        根据anchor和偏移量计算proposals
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = boxes.astype(deltas.dtype, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    # 计算任意多边形的面积，顶点按照顺时针或者逆时针方向排列
    def compute_polygon_area(self, points):
        point_num = len(points)
        if (point_num < 3):
            return 0.0
        s = points[0][1] * (points[point_num - 1][0] - points[1][0])
        # for i in range(point_num): # (int i = 1 i < point_num ++i):
        for i in range(1, point_num):  # 有小伙伴发现一个bug，这里做了修改，但是没有测试，需要使用的亲请测试下，以免结果不正确。
            s += points[i][1] * (points[i - 1][0] - points[(i + 1) % point_num][0])
        return abs(s / 2.0)

    def compute_polygon_area_numpy_list(self, point_numpy_list):
        for i in point_numpy_list:
            area = self.compute_polygon_area(i)

    # def rnms(self, dets, one_pixel=1):
    #     """
    #     nms,去除重复框
    #     """
    #     # x1、y1、x2、y2、以及score赋值
    #
    #     x1 = dets[:, 1]
    #     y1 = dets[:, 2]
    #     x2 = dets[:, 3]
    #     y2 = dets[:, 4]
    #     x3 = dets[:, 5]
    #     y3 = dets[:, 6]
    #     x4 = dets[:, 7]
    #     y4 = dets[:, 8]
    #     scores = dets[:, 0]
    #
    #     # 每一个检测框的面积
    #     areas = (0.5 * abs((x1 * y2 + x2 * y3 + x3 * y1) - (x1 * y3 + x2 * y1 + x3 * y2)) + one_pixel) * (
    #             0.5 * abs((x1 * y3 + x3 * y4 + x4 * y1) - (x1 * y4 + x3 * y1 + x4 * y3)) + one_pixel)
    #     # areas = (x2 - x1 + one_pixel) * (y2 - y1 + one_pixel)
    #     ## 按照score置信度降序排序
    #     order = scores.argsort()[::-1]
    #     keep = []  # 保留的结果框集合
    #     while order.size > 0:
    #         i = order[0]
    #         keep.append(i)  # 保留该类剩余box中得分最高的一个
    #         # 得到相交区域，左上及右下
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
    #
    #         # 计算相交面积，不重叠的时候为0
    #         w = np.maximum(0.0, xx2 - xx1 + one_pixel)
    #         h = np.maximum(0.0, yy2 - yy1 + one_pixel)
    #         inter = w * h
    #         # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
    #         ovr = inter / (areas[i] + areas[order[1:]] - inter)
    #         # 保留IoU小于阈值的box
    #         inds = np.where(ovr <= self.nms_thresh)[0]
    #         order = order[inds + 1]  # 因为ovr的数组长度比order数组少一个，所以这里要将所有下标后移一位
    #     return keep

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

    # def rnms(self, dets, one_pixel=1):
    #     """
    #     nms,去除重复框
    #     """
    #     # x1、y1、x2、y2、以及score赋值
    #     # x1 = dets[:, 1]
    #     # y1 = dets[:, 2]
    #     # x2 = dets[:, 3]
    #     # y2 = dets[:, 4]
    #     # x3 = dets[:, 5]
    #     # y3 = dets[:, 6]
    #     # x4 = dets[:, 7]
    #     # y4 = dets[:, 8]
    #     dets_bbox = []
    #     for i in dets:
    #         x1 = min(i[1], i[3], i[5], i[7])
    #         x2 = max(i[1], i[3], i[5], i[7])
    #         y1 = min(i[2], i[4], i[6], i[8])
    #         y2 = max(i[2], i[4], i[6], i[8])
    #         dets_bbox.append([x1, x2, y1, y2])
    #     dets_bbox = np.array(dets_bbox)
    #
    #     # x1 = dets[:, 1]
    #     # y1 = dets[:, 2]
    #     # x2 = dets[:, 3]
    #     # y2 = dets[:, 4]
    #     # x3 = dets[:, 5]
    #     # y3 = dets[:, 6]
    #     # x4 = dets[:, 7]
    #     # y4 = dets[:, 8]
    #     # scores = dets[:, 0]
    #     x1 = dets_bbox[:, 0]
    #     y1 = dets_bbox[:, 1]
    #     x2 = dets_bbox[:, 2]
    #     y2 = dets_bbox[:, 3]
    #     scores = dets[:, 0]
    #
    #     # 每一个检测框的面积
    #     # areas = (abs((x1 * y2 + x2 * y3 + x3 * y1) - (x1 * y3 + x2 * y1 + x3 * y2)) + one_pixel) * (
    #     #         abs((x1 * y3 + x3 * y4 + x4 * y1) - (x1 * y4 + x3 * y1 + x4 * y3)) + one_pixel)
    #     areas = (x2 - x1 + one_pixel) * (y2 - y1 + one_pixel)
    #     ## 按照score置信度降序排序
    #     order = scores.argsort()[::-1]
    #     keep = []  # 保留的结果框集合
    #     while order.size > 0:
    #         i = order[0]
    #         keep.append(i)  # 保留该类剩余box中得分最高的一个
    #         # 得到相交区域，左上及右下
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
    #
    #         # 计算相交面积，不重叠的时候为0
    #         w = np.maximum(0.0, xx2 - xx1 + one_pixel)
    #         h = np.maximum(0.0, yy2 - yy1 + one_pixel)
    #         inter = w * h
    #         # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
    #         ovr = inter / (areas[i] + areas[order[1:]] - inter)
    #         # 保留IoU小于阈值的box
    #         inds = np.where(ovr <= self.nms_thresh)[0]
    #         order = order[inds + 1]  # 因为ovr的数组长度比order数组少一个，所以这里要将所有下标后移一位
    #     return keep

    def nms(self, dets, one_pixel=1):
        """
        nms,去除重复框
        """
        # x1、y1、x2、y2、以及score赋值
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        # 每一个检测框的面积
        areas = (x2 - x1 + one_pixel) * (y2 - y1 + one_pixel)
        ## 按照score置信度降序排序
        order = scores.argsort()[::-1]
        keep = []  # 保留的结果框集合
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留该类剩余box中得分最高的一个
            # 得到相交区域，左上及右下
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算相交面积，不重叠的时候为0
            w = np.maximum(0.0, xx2 - xx1 + one_pixel)
            h = np.maximum(0.0, yy2 - yy1 + one_pixel)
            inter = w * h
            # 计算IoU：重叠面积/（面积1+面积2-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留IoU小于阈值的box
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]  # 因为ovr的数组长度比order数组少一个，所以这里要将所有下标后移一位
        return keep

    def _image_read(self, path):
        """
       利用rasterio读取数据并转成cv2读取图片的格式
       """
        with rasterio.open(path) as ds:
            img = ds.read()
            transform = ds.transform
            try:
                crs = ds.crs.data['init']
                crs = {"type": "name", "properties": {"name": crs}}
            except:
                crs = "null"

        img = img[:3, :, :]
        img = reshape_as_image(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return crs, img, transform

    def _view_bar(self, num, total):
        """
        进度条
        """
        rate = float(num) / float(total)
        rate_num = int(rate * 100)
        r = '\r[%s%s]%d%%,%d' % (">" * rate_num, "-" * (100 - rate_num), rate_num, num)
        sys.stdout.write(r)
        sys.stdout.flush()

    def write_origin_file(self, out_data, mult_categorys_nms, num_objects):
        with open(out_data, 'a') as file_out:
            for classe_name in self.classes:
                for temp_bbox in mult_categorys_nms[classe_name].split('\n'):
                    if temp_bbox.split() != []:
                        num_objects = num_objects + 1
                        outline = classe_name + ' ' + temp_bbox
                    file_out.write(outline + '\n')
