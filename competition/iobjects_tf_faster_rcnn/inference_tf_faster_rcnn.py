import os
import re

import yaml
from dotmap import DotMap
from faster_rcnn._detection import FasterRCNNEstimation


def get_config_from_yaml(yaml_file, encoding='utf8'):
    """
    Get the config from a yml or yaml file
    :param yaml_file: 文件路径
    :param encoding: encoding default: utf8
    :return: config(namespace) or config(dictionary)
    """
    with open(yaml_file, encoding=encoding) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config_dict)
    return config


def plane_detection(input_data, category_name, model_path, out_data, out_dataset_name,
                    nms_thresh,
                    score_thresh):
    # config = get_config_from_yaml(model_path)
    config_file = model_path
    model_path = os.path.abspath(os.path.join(model_path, os.path.pardir))
    if category_name is None:
        # 获取类别信息yaml
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config = DotMap(config_dict)
        # 模型中支持的所有类别
        config.get("model").get("categorys").remove("__background__")
        category_name = config.get("model").get("categorys")
        category_name = [str(i) for i in category_name]
        category_name = category_name
    else:
        regex = ",|，"
        category_name = re.split(regex, category_name)

    if not isinstance(model_path, str):
        raise TypeError('model_path must be str ')
    elif not os.path.exists(model_path):
        raise Exception('model_path does not exist ')

    if not isinstance(out_data, str):
        raise TypeError('out_data must be str ')
    if not isinstance(out_dataset_name, str):
        raise TypeError('out_dataset_name must be str ')

    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            run_prediction = FasterRCNNEstimation(model_path, config_file)
            run_prediction.estimation_dir(input_data, category_name,
                                          out_data, out_dataset_name,
                                          nms_thresh,
                                          score_thresh)

        else:
            run_prediction = FasterRCNNEstimation(model_path, config_file)
            run_prediction.estimation_img(input_data, category_name,
                                          out_data, out_dataset_name,
                                          nms_thresh,
                                          score_thresh)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    input_data = '/home/data/hou/workspaces/iobjectspy/resources_ml/example_data/training/plane.tif'
    model_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/model/obj_det_plane/obj_det_plane.sdm'
    out_data = os.path.join(curr_dir, 'out')
    out_name = 'plane'
    category_name = None
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    # 基于影像文件进行飞机目标检测
    plane_detection(input_data, category_name, model_path, out_data, out_name,
                    nms_thresh=0.3,
                    score_thresh=0.3)
