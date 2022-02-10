import os.path
from tkinter import Image

import cv2
import numpy as np

from iobjectspy import numpy_array_to_datasetvector
from iobjectspy.ml.toolkit._toolkit import to_onehot, get_config_from_yaml
from PIL import Image
curr_dir = os.path.dirname(os.path.abspath(__file__))



# def from_mask_get_rotating_rectangle(mask_data_path, sda_path,out_data_path):
#     data_config=get_config_from_yaml(sda_path)
#     class_type=data_config.dataset.class_type
#     # img = cv2.imread(mask_data_path)
#     mask = np.array(Image.open(mask_data_path))
#     if len(class_type) > 2:
#         mask = to_onehot(mask, [num for num in range(len(class_type))])
#     else:
#         mask = np.expand_dims(mask, -1)
#         # ds = get_output_datasource(output)
#     numpy_array_to_datasetvector(mask,out_data_path,'out_data_name')
#
#     # water = (mask == 1).astype(np.uint8)  # 01二值化
#     ret, mask = cv2.threshold(mask, 1, 3, cv2.THRESH_BINARY)
#     num_labels_w, labels_w, stats_w, centers_w = cv2.connectedComponentsWithStats(water,
#                                                                                   connectivity=8,
#                                                                                   ltype=cv2.CV_16U)
#     # water_connect_array = np.append(water_connect_array, num_labels_w)
#     # cnt = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # 必须是array数组的形式
#     # rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
#     # box = cv2.cv.BoxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
#     # box = np.int0(box)
#     # # 画出来
#     # cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
#     # cv2.imwrite('contours.png', img)

def from_mask_get_rotating_rectangle(mask_data_path, sda_path,out_data_path):
    data_config=get_config_from_yaml(sda_path)
    class_type=data_config.dataset.class_type
    # img = cv2.imread(mask_data_path)
    mask = np.array(Image.open(mask_data_path))
    if len(class_type) > 2:
        mask = to_onehot(mask, [num for num in range(len(class_type))])
    else:
        mask = np.expand_dims(mask, -1)
        # ds = get_output_datasource(output)
    numpy_array_to_datasetvector(mask,out_data_path,'out_data_name')

    # water = (mask == 1).astype(np.uint8)  # 01二值化
    ret, mask = cv2.threshold(mask, 1, 3, cv2.THRESH_BINARY)
    num_labels_w, labels_w, stats_w, centers_w = cv2.connectedComponentsWithStats(water,
                                                                                  connectivity=8,
                                                                                  ltype=cv2.CV_16U)
    # water_connect_array = np.append(water_connect_array, num_labels_w)
    # cnt = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # 必须是array数组的形式
    # rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    # box = cv2.cv.BoxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    # box = np.int0(box)
    # # 画出来
    # cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
    # cv2.imwrite('contours.png', img)


def from_point_dataset_get_rotating_rectangle(mask_data_path, sda_path, out_data_path):
    img = np.array(Image.open(mask_data_path))
    cnt = np.array([[101,101], [122, 140], [130, 150], [131, 157], [141, 140], [151, 130], [161, 120], [101, 101]])  # 必须是array数组的形式
    cv2.drawContours(img, [cnt], 0, (255, 0, 0), 1)
    cv2.imwrite('polygon.png', img)
    rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    box = np.int0(box)
    # 画出来
    cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
    cv2.imwrite('contours.png', img)


if __name__ == '__main__':
    mask_data_path = os.path.join(curr_dir, '..', '..', 'example_data', 'training', 'object_ext_train_data',
                                  'train_data', 'SegmentationObject', '00000000.png')
    sda_path=os.path.join(curr_dir, '..', '..', 'example_data', 'training', 'object_ext_train_data',
                                  'train_data', 'train_data.sda')
    out_data_path=os.path.join(curr_dir, '..', '..', 'out','temp.udbx')

    # from_mask_get_rotating_rectangle(mask_data_path,sda_path,out_data_path)
    point_dataset=mask_data_path
    from_point_dataset_get_rotating_rectangle(point_dataset, sda_path, out_data_path)

