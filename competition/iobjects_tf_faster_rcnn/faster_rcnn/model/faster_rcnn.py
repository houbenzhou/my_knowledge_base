# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from dotmap import DotMap

from ..datasets.factory import get_imdb
# import iobjectspy
from ..datasets.imdb import imdb as imdb1
from ..model.config import cfg, cfg_from_file, cfg_from_list
from ..model.test import test_net
from ..model.train_val import get_training_roidb, train_net
from ..nets.resnet_v1 import resnetv1
from ..utils.freeze_model import freeze_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# 获取图片数据的数量
def _load_image_set_number(train_data_path):
    """
    Load the indexes listed numbers in this dataset's image set file.
    """
    image_set_file = os.path.join(train_data_path, 'ImageSets', 'Main',
                                  'trainval.txt')
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return len(image_index)


def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        # print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        # print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = imdb1(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def train(train_data_path, config, epoch, batch_size, lr,
          log_path='',
          backbone_name="res101",
          backbone_weight_path='',
          output_model_path='',
          output_model_name='saved_model', pretrained_model_path=None):
    # 取所在文件的上一级目录
    # train_data_path = os.path.abspath(os.path.join(train_data_path, os.path.pardir))
    # ckpt模型文件保存文件夹
    ckpt_model_path = os.path.join(log_path, "ckpt_model_path")
    print('Output snapshot model will be saved to `{:s}`'.format(ckpt_model_path))

    if pretrained_model_path is not None:
        pretrained_model_path = os.path.join(pretrained_model_path, "ckpt_model_path")
        print('Pretrained model path will be saved to `{:s}`'.format(pretrained_model_path))

    # tensorboard 文件夹保存训练过程的日志
    tb_dir = os.path.join(log_path, "tensorflow")
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # pb 文件夹保存用于预测的模型文件
    pb_model_path = os.path.join(output_model_path, output_model_name)
    print('Output model will be saved to `{:s}`'.format(pb_model_path))
    # 若输出模型文件夹路径则创建输出模型文件夹路径
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    # 若日志路径不存在，则创建日志路径
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 若日志路径中ckpt存储文件夹不存在则创建ckpt文件夹
    if not os.path.exists(ckpt_model_path):
        os.makedirs(ckpt_model_path)
    # 若日志路径中tensorflow的log存储文件夹不存在则创建tensorflow文件夹
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    # 若模型文件输出路径不存在，则创建模型文件输出路径
    if not os.path.exists(pb_model_path):
        os.makedirs(pb_model_path)
    if not os.path.exists(backbone_weight_path):
        backbone_weight_path = None
    # 通过图片数量以及epoch计算出max_iters
    images_num = _load_image_set_number(train_data_path)
    max_iters = epoch * images_num
    # 通过配置文件传递参数
    cfg_from_file(config)
    # 通过VOC数据集中的.yml文件获取数据集的类别替换默认参数，通过参数替换默认参数
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.sda')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    voc_config = DotMap(config_dict)
    classes = voc_config.dataset.get('classes')
    classes_str = [str(i) for i in classes]
    cfg_from_list(
        ['TRAIN.IMS_PER_BATCH', batch_size, 'TRAIN.LEARNING_RATE', lr, 'TRAIN_DATA_PATH', train_data_path, 'CLASSES',
         classes_str])

    np.random.seed(cfg.RNG_SEED)
    # 训练数据集
    imdb, roidb = combined_roidb('voc_trainval')

    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb('voc_test')
    cfg.TRAIN.USE_FLIPPED = orgflip
    net = resnetv1(num_layers=101)

    output_model_name_ckpt = output_model_name + ".ckpt"

    # 训练ckpt模型格式的模型
    # train_net(net, imdb, roidb, valroidb, pretrained_model_path, ckpt_model_path, output_model_name_ckpt, tb_dir,
    #           backbone_weight_path,
    #           max_iters)
    # tf.reset_default_graph()
    anchor_scales = config.get("trainer").get('ANCHOR_SCALES')
    anchor_ratios = config.get("trainer").get('ANCHOR_RATIOS')

    # 将最后一个ckpt格式的模型转换为pb格式的模型文件
    freeze_model(os.path.join(ckpt_model_path, output_model_name_ckpt), net, pb_model_path,
                 len(voc_config.dataset.get('classes')) - 1, anchor_scales, anchor_ratios)
    # pb模型配置文件
    # 拼接字符串的方式获取目录名称
    # pb_yml_name = pb_model_path.split("/")[-1]
    # os自带的方法获取目录名称
    pic_names = os.listdir(os.path.join(train_data_path, 'Images'))
    im = Image.open(os.path.join(train_data_path, 'Images', pic_names[0]))
    blocksize = im.size[0]
    tile_offset = int(im.size[0] / 2)
    pb_yml_name = os.path.basename(pb_model_path)
    pb_config = os.path.join(pb_model_path, pb_yml_name + '.sdm')
    dict_detection = {'framework': 'tensorflow', 'model_type': 'detect', 'model_architecture': 'faster_rcnn',
                      'model_tag': 'standard', 'signature_name': 'predict',
                      'model': {'categorys': ["tree"], 'blocksize': blocksize, 'tile_offset': tile_offset}}
    dict_detection['model']['categorys'] = classes
    with open(pb_config, 'w', encoding='utf-8') as f:
        yaml.dump(dict_detection, f)
    print('model saved to `{:s}`'.format(os.path.join(pb_model_path)))


def test_evaluate(train_data_path, config, checkpoint_path, eval_path, max_per_image):
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    # 训练数据集
    filename = checkpoint_path
    filename[:-5]

    # 通过配置文件传递参数
    cfg_from_file(config)
    train_data_yml_name = os.path.basename(train_data_path)
    with open(os.path.join(train_data_path, train_data_yml_name + '.yml')) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    voc_config = DotMap(config_dict)
    classes = voc_config.get("CLASSES")
    cfg_from_list(
        ['TRAIN_DATA_PATH', train_data_path, 'CLASSES',
         classes])

    imdb = get_imdb('voc_test')
    imdb.competition_mode(False)

    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    print(('Loading model check point from {:s}').format(checkpoint_path))
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print('Loaded.')

    # 测试ckpt模型格式的模型
    test_net(sess, net, imdb, eval_path, max_per_image)
    tf.reset_default_graph()
    sess.close()
