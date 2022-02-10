

import yaml
import yaml
from dotmap import DotMap

import numpy as np

import os


def nms( dets, nms_thresh, one_pixel=1):
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
def modify_category(input_data,output_data,category_name,nms_thresh=0.3):
    """
    修改模型的推理结果

    :param input_data  模型推理结果保存路径
    :param output_data  通过一定的规则修正推理结果的保存路径

    """
    file_names = os.listdir(input_data)
    for file_name in file_names:
        input_file_path = os.path.join(input_data, file_name)
        input_file = open(input_file_path, "r", encoding="utf-8", errors="ignore")
        # one_file_all_boxs_list=[]
        all_boxes=[]
        out_all_bbox=[]
        pic_name=''
        try:
            while True:
                mystr = input_file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                list_bbox = mystr.split(',')
                pic_name = list_bbox[0]
                x = []
                y = []
                x.append(float(list_bbox[1]))
                x.append(float(list_bbox[3]))
                x.append(float(list_bbox[5]))
                x.append(float(list_bbox[7]))
                y.append(float(list_bbox[2]))
                y.append(float(list_bbox[4]))
                y.append(float(list_bbox[6]))
                y.append(float(list_bbox[8]))
                xmin = float(min(x))
                ymin = float(min(y))
                xmax = float(max(x))
                ymax = float(max(y))
                # 规则修改的位置
                score=float(list_bbox[10].strip())
                cls = list_bbox[9]

                zoom_out_bbox = []
                zoom_out_bbox.append(xmin)
                zoom_out_bbox.append(ymin)
                zoom_out_bbox.append(xmax)
                zoom_out_bbox.append(ymax)
                zoom_out_bbox.append(score)
                zoom_out_bbox.append(cls)
                all_boxes.append(zoom_out_bbox)
                # one_file_all_boxs_list.append(outline)

        except IOError:
            print(IOError)

        for cls_ind, cls in enumerate(category_name[0:]):
            all_boxes_temp=[]
            for i in all_boxes:
                if str(cls) == i[5]:
                    all_boxes_temp.append(i[0:5])
            all_boxes_temp = np.array(all_boxes_temp)
            if (all_boxes_temp != np.array([])):
                keep = nms(all_boxes_temp, nms_thresh, one_pixel=1)
                all_boxes_temp = all_boxes_temp[keep, :]
            for bbox_score in all_boxes_temp:
                list_box = []
                xmin = bbox_score[0]
                ymin = bbox_score[1]
                xmax = bbox_score[2]
                ymax = bbox_score[3]
                list_box.append(xmin)
                list_box.append(ymin)
                list_box.append(xmax)
                list_box.append(ymax)
                list_box.append(cls)
                list_box.append(bbox_score[4])
                list_box.append(pic_name)
                out_all_bbox.append(list_box)

        input_file.close()
        output_file_path = os.path.join(output_data, file_name)
        out_file = open(output_file_path, "w", encoding="utf-8", errors="ignore")
        for bbox in out_all_bbox:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            category_name=bbox[4]
            score=bbox[5]
            outline = str(pic_name) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(
                ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(
                category_name) + ',' + str(score)
            out_file.write(outline + '\n')



if __name__ == '__main__':
    input_data=r"E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer_merge"
    output_data=r"E:\workspaces\iobjectspy_master\resources_ml\competition\out\out_airport_test_infer_result"
    category_name=None
    sdm_path=r'E:\workspaces\iobjectspy_master\resources_ml\competition\train_inference_detectron2\out\qzbv3\baseline\model_epoch4\save_model\save_model.sdm'
    if not os.path.exists(output_data):
        os.mkdir(output_data)
    if category_name is None:
        # 获取类别信息yaml
        with open(sdm_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config = DotMap(config_dict)
        # 模型中支持的所有类别
        category_name = config.get("model").get("categorys")
        category_name = [str(i) for i in category_name]

    else:
        regex = ",|£¬"
        category_name = re.split(regex, category_name)
    modify_category(input_data,output_data,category_name)

