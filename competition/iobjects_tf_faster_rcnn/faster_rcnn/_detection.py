#!/usr/bin/env python
# encoding: utf-8
import logging
import os
import sys
import time

import cv2
import numpy as np
import rasterio
import tensorflow as tf
import yaml
from dotmap import DotMap
from rasterio import transform
from rasterio.plot import reshape_as_image
from rasterio.windows import Window

"""
影像数据目标检测
"""


class FasterRCNNEstimation(object):
    def __init__(self, model_path, cfg):
        self.model_path = model_path
        self.cfg = cfg
        self.load_model(model_path)

    def estimation_img(self, input_data, category_name, out_data, out_name, nms_thresh=0.3,
                       score_thresh=0.5):
        """
        进行影像数据目标检测
        """

        result, _ = self._estimation_img(input_data, category_name, os.path.join(out_data, out_name) + ".txt", out_name,
                                         nms_thresh,
                                         score_thresh)
        self.close_model(self.sess)
        return result

    def estimation_dir(self, input_data, category_name, out_data_path, out_name, nms_thresh=0.3,
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
        self.close_model(self.sess)

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

            all_boxes = []
            # 记录一个像素占据地理坐标的比率
            try:
                one_pixel = ds.res[0]
            except:
                one_pixel = 1

            if (ds.height <= self.blocksize) | (ds.width <= self.blocksize):
                all_boxes = self._get_bbox(ds, -1, -1,
                                           all_boxes)
                logging.info('()： Width and height less than or equal to ()'.format(self.input_data, self.blocksize))
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
                logging.info('()： Width and height is greater than ()'.format(self.input_data, self.blocksize))

            # 对all_boxes中所有的框整体去重
            num_objects = 0
            with open(out_data, 'a') as file_out:
                for cls_ind, cls in enumerate(self.category_name[0:]):
                    all_boxes_temp = []
                    for i in all_boxes:
                        if str(cls) == i[5]:
                            all_boxes_temp.append(i[0:5])
                    all_boxes_temp = np.array(all_boxes_temp)
                    if (all_boxes_temp != np.array([])):
                        keep = self.nms(all_boxes_temp, one_pixel)
                        all_boxes_temp = all_boxes_temp[keep, :]
                        num_objects = len(all_boxes_temp)
                    for bbox_sore in all_boxes_temp:
                        outline = cls + ' ' + out_name + ' ' + str((int(bbox_sore[4] * 10000) / 10000)) + ' ' + str(
                            (int(bbox_sore[0] * 100) / 100)) + ' ' + str(
                            (int(bbox_sore[1] * 100) / 100)) + ' ' + str((int(bbox_sore[2] * 100) / 100)) + ' ' + str(
                            (int(bbox_sore[3] * 100) / 100))
                        file_out.write(outline + '\n')

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
        """
        处理每个tile输入模型的返回结果
        """
        transf = ds.transform

        height = ds.height
        width = ds.width

        try:
            one_pixel = ds.res[0]
        except:
            one_pixel = 1
        block_xmin = j * self.tile_offset
        block_ymin = i * self.tile_offset
        if (j == -1) & (i == -1):
            block_xmin = 0
            block_ymin = 0
            block = np.zeros([3, ds.height, ds.width], dtype=np.uint8)
            img = ds.read(window=Window(block_xmin, block_ymin, ds.width, ds.height))

        else:
            block_xmax = block_xmin + self.blocksize
            block_ymax = block_ymin + self.blocksize
            if height <= block_ymax:
                block_ymin = height - self.blocksize
            if width <= block_xmax:
                block_xmin = width - self.blocksize
            block = np.zeros([3, self.blocksize, self.blocksize], dtype=np.uint8)
            img = ds.read(window=Window(block_xmin, block_ymin, self.blocksize, self.blocksize))

        block[:, :img.shape[1], :img.shape[2]] = img[:3, :, :]
        block = reshape_as_image(block)
        block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)

        blobs, im_scales = self._get_blobs(block)
        im_blob = blobs['data']
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]],
                                    dtype=np.float32)
        # 执行预测
        _, scores, bbox_pred, rois = self.sess.run([self.score, self.prob, self.pred, self.rois],
                                                   feed_dict={self.im_data: blobs['data'],
                                                              self.im_info: blobs[
                                                                  'im_info']})
        self.sess.graph.finalize()
        # 计算包围框
        boxes = rois[:, 1:5] / im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])

        if True:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = self._bbox_transform_inv(boxes, box_deltas)
            pred_boxes = self._clip_boxes(pred_boxes, block.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        # # 遍历预测出的所有类别
        for cls_ind, cls in enumerate(self.category_names[0:]):
            cls_ind += 1  # 过滤了背景框
            cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = self.nms(dets, one_pixel)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= self.score_thresh)[0]
            # 将准确率大于阈值且用户需要的类别结果提取保存出来
            if len(inds) > 0 and str(cls) in self.category_name:
                # 输入为影像文件时，输出GeoJSON文件
                for s in inds:
                    bbox = dets[s, :5]
                    if ds.crs is None:
                        xmin = round(float(bbox[0]), 4) + block_xmin
                        ymin = round(float(bbox[3]), 4) + block_ymin
                        xmax = round(float(bbox[2]), 4) + block_xmin
                        ymax = round(float(bbox[1]), 4) + block_ymin
                        score_single_bbox = round(float(bbox[4]), 4)
                    else:
                        coord_min = transform.xy(transf, bbox[1] + float(block_ymin),
                                                 bbox[0] + float(block_xmin))
                        coord_max = transform.xy(transf, bbox[3] + float(block_ymin),
                                                 bbox[2] + float(block_xmin))

                        xmin = coord_min[0]
                        ymin = coord_max[1]
                        xmax = coord_max[0]
                        ymax = coord_min[1]
                        score_single_bbox = bbox[4]
                    all_boxes.append(
                        [xmin, ymin, xmax, ymax, score_single_bbox, str(cls)])

        return all_boxes

    def close_model(self, sess):
        """
        关闭模型
        """
        sess.close()
        tf.reset_default_graph()

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

    def load_model(self, model_path):
        """
       加载模型文件
       """
        self.model_path = model_path
        self.sess = tf.Session()
        meta_graph_def = tf.saved_model.loader.load(self.sess, ['serve'], model_path)
        signature = meta_graph_def.signature_def
        # 通过标记获取输入输出的张量名称
        im_data_tensor_name = signature['predict'].inputs['im_data'].name
        im_info_tensor_name = signature['predict'].inputs['im_info'].name
        score_tensor_name = signature['predict'].outputs['score'].name
        prob_tensor_name = signature['predict'].outputs['prob'].name
        pred_tensor_name = signature['predict'].outputs['pred'].name
        rois_tensor_name = signature['predict'].outputs['rois'].name
        # 获取输入的张量
        self.im_data = self.sess.graph.get_tensor_by_name(im_data_tensor_name)
        self.im_info = self.sess.graph.get_tensor_by_name(im_info_tensor_name)
        # 获取输出节点的张量
        self.score = self.sess.graph.get_tensor_by_name(score_tensor_name)
        self.prob = self.sess.graph.get_tensor_by_name(prob_tensor_name)
        self.pred = self.sess.graph.get_tensor_by_name(pred_tensor_name)
        self.rois = self.sess.graph.get_tensor_by_name(rois_tensor_name)
        # 获取类别信息yaml
        with open(self.cfg) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config = DotMap(config_dict)
        # 模型中支持的所有类别
        config.get("model").get("categorys").remove("__background__")
        self.category_names = config.get("model").get("categorys")
        # 切图时每个图块的尺寸
        self.blocksize = config.get("model").get("blocksize")
        # 向左下方移动的图块尺寸
        self.tile_offset = config.get("model").get("tile_offset")
        # 模型中支持的类别总数+1（加1是指的背景）
        config_categorys_num = len(config.get("model").get("categorys")) + 1
        tf.reset_default_graph()
        stds = np.tile(np.array([0.1, 0.1, 0.2, 0.2]), config_categorys_num)
        means = np.tile(np.array([0.0, 0.0, 0.0, 0.0]), config_categorys_num)

        self.pred *= stds
        self.pred += means

        if self.tile_offset == self.blocksize:
            logging.error('tile_offset and blocksize is same!')
            raise ValueError('tile_offset and blocksize is same!')
        elif self.tile_offset > self.blocksize:
            logging.error('tile_offset is bigger than blocksize!')
            raise ValueError('tile_offset is bigger than blocksize!')
