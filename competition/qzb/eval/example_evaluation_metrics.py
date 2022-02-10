# !/usr/bin/env python3
# coding=utf-8
import os
import time
import warnings
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from iobjectspy import open_datasource
from iobjectspy.ml.vision._evaluations import Evaluation

curr_dir = os.path.dirname(os.path.abspath(__file__))
resource_ml_dir = os.path.join(os.path.dirname(curr_dir))
data_dir = os.path.join(resource_ml_dir, 'example_data', 'eval')
model_path = os.path.join(resource_ml_dir, 'model')
out_dir = os.path.join(curr_dir, '..', 'out')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 检测指标计算
def example_obj_metrics(y_true,y_pred,class_name):
    # 影像目标检测指标计算
    # metrics_udbx = os.path.join(data_dir, 'obj_det', 'obj_det_img.udbx')
    # ds_det = open_datasource(metrics_udbx)
    # y_true = ds_det['out_plane_R']
    # y_pred = ds_det['out_plane1_R']
    # class_name = os.path.join(model_path, 'obj_det_plane', 'obj_det_plane.sdm')
    # true_field_name = 'category'
    # print("影像目标检测指标评价：")
    #
    # start_time = time.time()
    # mean_ap, eval_results = Evaluation.obj_vector_mean_ap(y_true, y_pred, true_field_name=true_field_name,
    #                                                       predict_field_name='category', class_name=class_name,
    #                                                       iou_thr=0.5)
    # print('完成，共耗时{}s，影像目标检测基于datasetVector计算mAP: {}'.format(
    #     time.time() - start_time, mean_ap))
    #
    # start_time = time.time()
    # recalls, precision, eval_results = Evaluation.obj_vector_prec_rec(y_true, y_pred, true_field_name=true_field_name,
    #                                                                   predict_field_name='category',
    #                                                                   class_name=class_name,
    #                                                                   iou_thr=0.5)
    # print('完成，共耗时{}s，影像目标检测基于datasetVector计算recalls: {},precision: {}'.format(
    #     time.time() - start_time, recalls, precision))
    #
    # start_time = time.time()
    # f1, eval_results = Evaluation.obj_vector_f1(y_true, y_pred, true_field_name=true_field_name,
    #                                             predict_field_name='category', class_name=class_name,
    #                                             iou_thr=0.5)
    # print('完成，共耗时{}s，影像目标检测基于datasetVector计算f1: {}'.format(
    #     time.time() - start_time, f1))

    # 图片目标检测指标评价
    # y_true = os.path.join(data_dir, 'obj_det', 'obj_det_pic')
    # y_pred = os.path.join(data_dir, 'obj_det', 'obj_det_pic')
    # class_name = os.path.join(model_path, 'obj_det_plane', 'obj_det_plane.sdm')
    print("图片目标检测指标评价")

    start_time = time.time()
    mean_ap, _ = Evaluation.obj_xml_mean_ap(y_true, y_pred, class_name=class_name,
                                            iou_thr=0.5)
    print('完成，共耗时{}s，图片目标检测基于xml文件计算mAP: {}'.format(
        time.time() - start_time, mean_ap))

    start_time = time.time()
    recalls, precision, _ = Evaluation.obj_xml_prec_rec(y_true, y_pred, class_name=class_name,
                                                        iou_thr=0.5)
    print('完成，共耗时{}s，图片目标检测基于xml文件计算recalls: {},precision: {}'.format(
        time.time() - start_time, recalls, precision))

    start_time = time.time()
    f1, _ = Evaluation.obj_xml_f1(y_true, y_pred, class_name=class_name,
                                  iou_thr=0.5)
    print('完成，共耗时{}s，图片目标检测基于xml文件计算f1: {}'.format(
        time.time() - start_time, f1))


# 地物分类指标计算

def example_seg_metrics():
    metrics_data = os.path.join(data_dir, 'seg_metrics')
    metrics_udb = os.path.join(data_dir, 'seg_metrics', 'metrics_test.udb')
    ds = open_datasource(metrics_udb)
    y_true_mc_v = ds['y_true_mc_v']
    y_predict_mc_v = ds['y_predict_mc_v']
    y_true_bin_v = ds['y_true_bin_v']
    y_predict_bin_v = ds['y_predict_bin_v']

    y_true_bin_tif = os.path.join(metrics_data, 'y_true_bin.tif')
    y_predict_bin_tif = os.path.join(metrics_data, 'y_predict_bin.tif')
    y_true_mc_tif = os.path.join(metrics_data, 'y_true_mc.tif')
    y_predict_mc_tif = os.path.join(metrics_data, 'y_predict_mc.tif')

    # bin
    print('二元分类指标评价')
    iou_macro = Evaluation.seg_iou(y_true_bin_tif, y_predict_bin_tif)
    f1_macro = Evaluation.seg_f1(y_true_bin_tif, y_predict_bin_tif)
    dice_macro = Evaluation.seg_dice(y_true_bin_tif, y_predict_bin_tif)
    kappa_macro = Evaluation.seg_kappa(y_true_bin_tif, y_predict_bin_tif)
    pixel_macro = Evaluation.seg_pixelacc(y_true_bin_tif, y_predict_bin_tif)

    print('二元分类基于栅格指标计算结果')
    print('iou : {},f1 : {},dice : {},kappa : {},pixel : {},'.format(iou_macro, f1_macro,
                                                                     dice_macro, kappa_macro,
                                                                     pixel_macro))
    cm = Evaluation.seg_confusion_matrix(y_true_bin_tif, y_predict_bin_tif)

    print('confusion_matrix_by_raster: ')
    print(cm)

    iou_macro = Evaluation.seg_iou(y_true_bin_v, y_predict_bin_v, 'value', 'value')
    f1_macro = Evaluation.seg_f1(y_true_bin_v, y_predict_bin_v, 'value', 'value', )
    dice_macro = Evaluation.seg_dice(y_true_bin_v, y_predict_bin_v, 'value', 'value')
    kappa_macro = Evaluation.seg_kappa(y_true_bin_v, y_predict_bin_v, 'value', 'value', )
    pixel_macro = Evaluation.seg_pixelacc(y_true_bin_v, y_predict_bin_v, 'value', 'value')

    print('二元分类基于矢量指标计算结果')
    print('iou {},f1 : {},dice : {},kappa : {},pixel : {},'.format(iou_macro, f1_macro,
                                                                   dice_macro, kappa_macro,
                                                                   pixel_macro))
    cm = Evaluation.seg_confusion_matrix(y_true_bin_v, y_predict_bin_v, 'value', 'value', )

    print('confusion_matrix_by_vector: ')
    print(cm)

    # 多分类指标计算,需要选择不同类别指标的平均方式,micro不区分类别计算所有像素或区域的指标，macro分类计算出指标后然后进行平均
    print('多分类:')
    iou_macro = Evaluation.seg_iou(y_true_mc_tif, y_predict_mc_tif, average='macro')
    f1_macro = Evaluation.seg_f1(y_true_mc_tif, y_predict_mc_tif, average='macro')
    dice_macro = Evaluation.seg_dice(y_true_mc_tif, y_predict_mc_tif, average='macro')
    kappa_macro = Evaluation.seg_kappa(y_true_mc_tif, y_predict_mc_tif)
    pixel_macro = Evaluation.seg_pixelacc(y_true_mc_tif, y_predict_mc_tif, average='macro')

    print('多分类基于栅格指标计算结果(macro):')
    print('iou_macro: {},f1_macro: {},dice_macro: {},kappa_macro: {},pixel_macro: {},'.format(iou_macro, f1_macro,
                                                                                              dice_macro, kappa_macro,
                                                                                              pixel_macro))

    iou_macro = Evaluation.seg_iou(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='macro')
    f1_macro = Evaluation.seg_f1(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='macro')
    dice_macro = Evaluation.seg_dice(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='macro')
    kappa_macro = Evaluation.seg_kappa(y_true_mc_v, y_predict_mc_v, 'value', 'value', )
    pixel_macro = Evaluation.seg_pixelacc(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='macro')

    print('多分类基于矢量指标计算结果(macro):')
    print('iou_macro: {},f1_macro: {},dice_macro: {},kappa_macro: {},pixel_macro: {},'.format(iou_macro, f1_macro,
                                                                                              dice_macro, kappa_macro,
                                                                                              pixel_macro))

    iou_micro = Evaluation.seg_iou(y_true_mc_tif, y_predict_mc_tif, average='micro')
    f1_micro = Evaluation.seg_f1(y_true_mc_tif, y_predict_mc_tif, average='micro')
    dice_micro = Evaluation.seg_dice(y_true_mc_tif, y_predict_mc_tif, average='micro')
    kappa_micro = Evaluation.seg_kappa(y_true_mc_tif, y_predict_mc_tif)
    pixel_micro = Evaluation.seg_pixelacc(y_true_mc_tif, y_predict_mc_tif, average='micro')
    cm = Evaluation.seg_confusion_matrix(y_true_mc_tif, y_predict_mc_tif)
    print('多分类基于栅格指标计算结果(micro):')
    print('iou_micro: {},f1_micro: {},dice_micro: {},kappa_micro: {},pixel_micro: {},'.format(iou_micro, f1_micro,
                                                                                              dice_micro, kappa_micro,
                                                                                              pixel_micro))
    print('confusion_matrix: ')
    print(cm)

    iou_micro = Evaluation.seg_iou(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='micro')
    f1_micro = Evaluation.seg_f1(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='micro')
    dice_micro = Evaluation.seg_dice(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='micro')
    kappa_micro = Evaluation.seg_kappa(y_true_mc_v, y_predict_mc_v, 'value', 'value', )
    pixel_micro = Evaluation.seg_pixelacc(y_true_mc_v, y_predict_mc_v, 'value', 'value', average='micro')
    cm = Evaluation.seg_confusion_matrix(y_true_mc_v, y_predict_mc_v, 'value', 'value')
    print('多分类基于矢量指标计算结果(micro):')
    print('iou_micro: {},f1_micro: {},dice_micro: {},kappa_micro: {},pixel_micro: {},'.format(iou_micro, f1_micro,
                                                                                              dice_micro, kappa_micro,
                                                                                              pixel_micro))
    print('confusion_matrix: ')
    print(cm)


if __name__ == '__main__':
    y_true = r'E:\workspaces\iobjectspy_master\resources_ml\out\picture_cascade_rcnn_plane\VOC\Annotations'
    y_pred = r'E:\workspaces\iobjectspy_master\resources_ml\out\picture_cascade_rcnn_plane\out_result_data'
    class_name = r'E:\workspaces\iobjectspy_master\resources_ml\out\picture_cascade_rcnn_plane\VOC\VOC.sda'
    # print("图片目标检测指标评价")
    example_obj_metrics(y_true,y_pred,class_name)
    # example_seg_metrics()
