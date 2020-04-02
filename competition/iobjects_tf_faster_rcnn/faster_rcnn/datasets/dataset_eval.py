# !/usr/bin/env python3
# coding=utf-8
import os

from iobjectspy import Rectangle, intersect


def dataset_eval(inference_data,
                 gt_data,
                 classname,
                 ovthresh=0.5,
                 out_path=False):
    """rec, prec = dataset_eval(inference_data,
                 gt_data,
                 classname,
                 ovthresh=0.5,
                 out_path=False)

    评估矢量数据集的

    inference_data: 目标检测预测返回的矢量数据集结果
    gt_data: 标签数据集
    classname: 类别名称
    [ovthresh]: Overlap threshold (default = 0.5)
    out_path:评估结果输出路径，如果不输入则默认不保存
    """
    inference_count = inference_data.get_record_count()
    gt_count = gt_data.get_record_count()

    rectangle_gt = gt_data.bounds
    rectangle_inference = inference_data.bounds
    left = max(int(rectangle_gt.left), int(rectangle_inference.left))
    right = min(int(rectangle_gt.right), int(rectangle_inference.right))
    top = min(int(rectangle_gt.top), int(rectangle_inference.top))
    bottom = max(int(rectangle_gt.bottom), int(rectangle_inference.bottom))

    rectangle = Rectangle(left, top, right, bottom)
    tp = 0
    i = 0
    recordset_inference = inference_data.query_with_bounds(rectangle, cursor_type='STATIC')
    # recordset_gt = gt_data.query_with_bounds(rectangle, cursor_type='STATIC')

    if recordset_inference.get_geometry() is not None:
        for feature_inference in recordset_inference.get_features():

            feature_area = feature_inference.geometry.area
            bounds_inference = feature_inference.geometry.bounds

            left = bounds_inference.left
            right = bounds_inference.right
            top = bounds_inference.top
            bottom = bounds_inference.bottom

            tile_box = Rectangle(left,
                                 bottom,
                                 right,
                                 top)
            recordset_gt = gt_data.query_with_bounds(tile_box, cursor_type='STATIC')
            if recordset_gt.get_geometry() is not None:
                for feature_gt in recordset_gt.get_features():
                    iou_geometry = intersect(feature_inference.geometry, feature_gt.geometry)
                    if iou_geometry == None:
                        iou_area = 0
                    else:
                        iou_area = iou_geometry.area
                    if iou_area > (ovthresh * (feature_area + feature_gt.geometry.area - iou_area)):
                        tp = tp + 1
            i = i + 1
            # 创建输出路径
        if out_path == False:
            pass
        else:
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
        fp = inference_count - tp
        fn = gt_count - tp

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
    return prec, rec
