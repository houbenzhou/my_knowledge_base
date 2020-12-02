import argparse
import os
import re

import yaml
from dotmap import DotMap
from faster_rcnn._detection import FasterRCNNEstimation


def get_config_from_yaml(yaml_file, encoding='utf8'):
    """
    Get the configs from a yml or yaml file
    :param yaml_file: 文件路径
    :param encoding: encoding default: utf8
    :return: configs(namespace) or configs(dictionary)
    """
    with open(yaml_file, encoding=encoding) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config_dict)
    return config


def plane_detection(input_data, category_name, model_path, out_data, out_dataset_name,
                    nms_thresh,
                    score_thresh):
    # configs = get_config_from_yaml(model_path)
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
                                          out_data,
                                          nms_thresh,
                                          score_thresh)

        else:
            run_prediction = FasterRCNNEstimation(model_path, config_file)
            run_prediction.estimation_img(input_data, category_name,
                                          out_data, out_dataset_name,
                                          nms_thresh,
                                          score_thresh)


def get_parser():
    parser = argparse.ArgumentParser(description="tf_faster_rcnn infer")
    parser.add_argument(
        "--input_data",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/dota/dotav1_test/images",
        help="A file or directory of input data",
    )

    parser.add_argument(
        "--model_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/iobjects_tf_faster_rcnn/out/plane/model/saved_model/saved_model.sdm',
        help="A file of model path",
    )

    parser.add_argument(
        "--out_data",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/iobjects_tf_faster_rcnn/out',
        help="A directory to save the output inference file. ",
    )
    parser.add_argument(
        "--out_name",
        default='plane',
        help="When input_data is a file,out_name is the name of the txt file ",
    )
    parser.add_argument(
        "--category_name",
        default=None,
        help="The name of the category you want to predict",
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.3,
        help="The threshold of nms,used to deal with the problem of target overlap",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.3,
        help="The threshold of score,used to filter low score targets ",
    )

    return parser


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    args = get_parser().parse_args()

    input_data = args.input_data
    model_path = args.model_path
    out_data = args.out_data
    out_name = args.out_name
    category_name = args.category_name
    nms_thresh = args.nms_thresh
    score_thresh = args.score_thresh
    if not os.path.exists(out_data):
        os.makedirs(out_data)
    # 基于影像文件进行飞机目标检测
    plane_detection(input_data, category_name, model_path, out_data, out_name,
                    nms_thresh=nms_thresh,
                    score_thresh=score_thresh)
