import argparse
import os
import random

import numpy as np
import rasterio
from iobjectspy_tools import register_all_pascal_voc, get_classname, get_class_num
from numpy import linspace
from rasterio.plot import reshape_as_image
from rasterio.windows import Window, transform

from detectron2.utils.logger import setup_logger

setup_logger()

import cv2

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def inference_detectron2(train_data_path, train_config_path, image_path, register_val_name, model_path, outpath):
    cfg = get_cfg()
    cfg.merge_from_file(train_config_path)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_path  # initialize from model zoo
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    num_class = get_class_num(train_data_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class  # get classes from sda

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.DATASETS.TEST = (register_val_name,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    pic_names = os.listdir(image_path)
    for d in pic_names:
        _estimation_img(os.path.join(image_path, d), os.path.join(outpath, str(d).split('.')[0] + ".txt"),
                        str(d).split('.')[0], 600, 300, predictor)


def _estimation_img(input_data, out_data, out_name, blocksize, tile_offset, predictor, nms_thresh=0.3,
                    score_thresh=0.3):
    """
    进行影像数据目标检测
    """

    with rasterio.open(input_data) as ds:
        width_block = ds.width // tile_offset + 1
        height_block = ds.height // tile_offset + 1

        all_boxes = []
        # 记录一个像素占据地理坐标的比率
        try:
            one_pixel = ds.res[0]
        except:
            one_pixel = 1

        if (ds.height <= blocksize) | (ds.width <= blocksize):
            all_boxes = _get_bbox(ds, -1, -1, blocksize, tile_offset, predictor,
                                  all_boxes)

        else:
            # 记录程序运行进度条
            p = 0
            for i in range(height_block):
                for j in range(width_block):
                    all_boxes = _get_bbox(ds, j, i, blocksize, tile_offset, predictor,
                                          all_boxes)
                    p += 1
                    # self._view_bar(p, (height_block) * (width_block))

        # 对all_boxes中所有的框整体去重
        num_objects = 0
        category_name = linspace(1, 16, 16)
        with open(out_data, 'a') as file_out:
            print(out_data)
            for cls_ind, cls in enumerate(category_name[0:]):
                all_boxes_temp = []
                for i in all_boxes:
                    if str(int(cls)) == i[5]:
                        all_boxes_temp.append(i[0:5])
                all_boxes_temp = np.array(all_boxes_temp)
                if (all_boxes_temp != np.array([])):
                    keep = nms(all_boxes_temp, nms_thresh, one_pixel)
                    all_boxes_temp = all_boxes_temp[keep, :]
                    num_objects = len(all_boxes_temp)
                for bbox_sore in all_boxes_temp:
                    outline = str(int(cls)) + ' ' + out_name + ' ' + str(bbox_sore[4]) + ' ' + str(
                        bbox_sore[0]) + ' ' + str(
                        bbox_sore[1]) + ' ' + str(bbox_sore[2]) + ' ' + str(bbox_sore[3])
                    file_out.write(outline + '\n')

    return 1, num_objects


def nms(dets, nms_thresh, one_pixel=1):
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
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]  # 因为ovr的数组长度比order数组少一个，所以这里要将所有下标后移一位
    return keep


def _get_bbox(ds, j, i, blocksize, tile_offset, predictor,
              all_boxes):
    """
    处理每个tile输入模型的返回结果
    """
    transf = ds.transform

    height = ds.height
    width = ds.width

    # try:
    #     one_pixel = ds.res[0]
    # except:
    #     one_pixel = 1
    block_xmin = j * tile_offset
    block_ymin = i * tile_offset
    if (j == -1) & (i == -1):
        block_xmin = 0
        block_ymin = 0
        block = np.zeros([3, ds.height, ds.width], dtype=np.uint8)
        img = ds.read(window=Window(block_xmin, block_ymin, ds.width, ds.height))

    else:
        block_xmax = block_xmin + blocksize
        block_ymax = block_ymin + blocksize
        if height <= block_ymax:
            block_ymin = height - blocksize
        if width <= block_xmax:
            block_xmin = width - blocksize
        block = np.zeros([3, blocksize, blocksize], dtype=np.uint8)
        img = ds.read(window=Window(block_xmin, block_ymin, blocksize, blocksize))

    block[:, :img.shape[1], :img.shape[2]] = img[:3, :, :]
    block = reshape_as_image(block)
    block = cv2.cvtColor(block, cv2.COLOR_RGB2BGR)

    outputs = predictor(block)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    pred_classes = outputs["instances"].pred_classes.to("cpu").numpy()
    scores = outputs["instances"].scores.to("cpu").numpy()
    pred_boxes = outputs["instances"].pred_boxes.to("cpu").tensor.numpy()
    # # 遍历预测出的所有类别
    for cls_ind, cls in enumerate(pred_classes):

        if ds.crs is None:
            xmin = round(float(pred_boxes[cls_ind][0]), 4) + block_xmin
            ymin = (round(float(pred_boxes[cls_ind][3]), 4) + block_ymin)
            xmax = round(float(pred_boxes[cls_ind][2]), 4) + block_xmin
            ymax = (round(float(pred_boxes[cls_ind][1]), 4) + block_ymin)
            score_single_bbox = round(float(scores[cls_ind]), 4)
        else:
            coord_min = transform.xy(transf, pred_boxes[cls_ind][1] + float(block_ymin),
                                     pred_boxes[cls_ind][0] + float(block_xmin))
            coord_max = transform.xy(transf, pred_boxes[cls_ind][3] + float(block_ymin),
                                     pred_boxes[cls_ind][2] + float(block_xmin))

            xmin = coord_min[0]
            ymin = coord_max[1]
            xmax = coord_max[0]
            ymax = coord_min[1]
            score_single_bbox = scores[cls_ind]
        all_boxes.append(
            [xmin, ymin, xmax, ymax, score_single_bbox, str(cls)])

    return all_boxes


def get_parser():
    parser = argparse.ArgumentParser(description="train detectron2")
    parser.add_argument(
        "--train_data_path",
        default="/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC",
        help="path to train data directory",
    )

    parser.add_argument(
        "--train_config_path",
        default='/home/data/hou/workspaces/detectron2/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml',
        help="path to config file",
    )

    parser.add_argument(
        "--input_image_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images',
        help="path to input image data directory ",
    )
    parser.add_argument(
        "--model_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/temp/model_final.pth',
        help="path to model path directory ",
    )
    parser.add_argument(
        "--outpath",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/visual',
        help="path to output directory ",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    train_data_path = args.train_data_path
    train_config_path = args.train_config_path
    input_image_path = args.input_image_path
    model_path = args.model_path
    outpath = args.outpath

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    data_path_name = train_data_path.split("/")[-1]

    class_names = get_classname(train_data_path)

    register_all_pascal_voc(train_data_path=train_data_path, class_names=class_names,
                            )
    register_val_name = data_path_name + '_test'

    inference_detectron2(train_data_path, train_config_path, input_image_path, register_val_name, model_path,
                         outpath)
