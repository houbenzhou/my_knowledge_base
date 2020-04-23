import os
import time

import yaml
from dotmap import DotMap
from faster_rcnn.model import faster_rcnn
from easydict import EasyDict as edict
curr_dir = os.path.dirname(os.path.abspath(__file__))
backbone_weight_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/backbone/res101.ckpt'

backbone_name = 'res101'

config1 = '/home/data/hou/workspaces/iobjectspy/resources_ml/trainer_config/object_detection_train_config.sdt'


def get_config_from_yaml(yaml_file, encoding='utf8'):
    """
    Get the config from a yml or yaml file
    :param yaml_file: 文件路径
    :param encoding: encoding default: utf8
    :return: config(namespace) or config(dictionary)
    """
    with open(yaml_file, encoding=encoding) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = edict(config_dict)
    return config


config = get_config_from_yaml(config1)


def example_train():
    """
    利用VOC数据集训练模型文件.
    """
    epoch = 1
    batch_size = 1
    lr = 0.001,

    train_data_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/out/2020-04-13_/光3_512_modify_1_3/VOC'
    log_path = os.path.join(curr_dir, 'out', 'log')
    output_model_path = os.path.join(curr_dir, 'out', 'model')
    output_model_name = 'saved_model'

    start_time = time.time()

    faster_rcnn.train(train_data_path, config, epoch, batch_size, lr[0],
                      log_path, backbone_name, backbone_weight_path, output_model_path,
                      output_model_name='saved_model', pretrained_model_path=None)
    print('完成，共耗时{}s，模型文件保存在 {}'.format(
        time.time() - start_time, os.path.join(output_model_path, output_model_name)))


if __name__ == '__main__':
    example_train()
