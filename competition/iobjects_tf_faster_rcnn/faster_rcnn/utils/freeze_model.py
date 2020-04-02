#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil

from ..model.config import cfg
from ..model.test import im_detect
import tensorflow as tf
import os

from ..nets.resnet_v1 import resnetv1

slim = tf.contrib.slim
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model import builder as saved_model_builder


def freeze_model(input_checkpoint, net, output_graph, num_category):
    """
        模型文件转换，将训练得到的ckpt格式的模型文件转化为用以iobjectspy组件的产品

        :param input_checkpoint: 输入训练好的ckpt模型文件
        :type input_checkpoint: str
        :param net: faster-rcnn用到的基础网络,"vgg16"或者"resnet101"
        :type net: str
        :param output_graph: 输出pb模型的路径
        :type output_graph: str
        :param num_category: 模型支持的类别数量
        :type num_category: int

        """
    # cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    if not os.path.isfile(input_checkpoint + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(input_checkpoint + '.meta'))
    if not os.path.exists(output_graph):
        os.makedirs(output_graph)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", num_category + 1,
                            tag='default', anchor_scales=[8, 16, 32])
    output_node_names = "resnet_v1_101_5/cls_score/BiasAdd,resnet_v1_101_5/cls_prob,resnet_v1_101_5/bbox_pred/BiasAdd,resnet_v1_101_3/rois/concat"
    output_graph_temp = output_graph + "temp.pb"
    saver = tf.train.Saver()
    saver.restore(sess, input_checkpoint)
    # transfer model
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=output_node_names.split(","))
    with tf.gfile.GFile(output_graph_temp, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()
    tf.reset_default_graph()
    export_tfserving_model(output_graph_temp, output_graph)


def export_tfserving_model(output_graph_temp, output_graph):
    export_path = os.path.join(output_graph)
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(output_graph_temp, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with tf.Session(graph=detection_graph) as sess:
        signature = predict_signature_def(inputs={'im_data': sess.graph.get_tensor_by_name('Placeholder:0'),
                                                  'im_info': sess.graph.get_tensor_by_name('Placeholder_1:0')},
                                          outputs={'score': sess.graph.get_tensor_by_name(
                                              'resnet_v1_101_5/cls_score/BiasAdd:0'),
                                              'prob': sess.graph.get_tensor_by_name('resnet_v1_101_5/cls_prob:0'),
                                              'pred': sess.graph.get_tensor_by_name(
                                                  'resnet_v1_101_5/bbox_pred/BiasAdd:0'),
                                              'rois': sess.graph.get_tensor_by_name(
                                                  'resnet_v1_101_3/rois/concat:0')})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()
    # sess.close()
    tf.reset_default_graph()

    if os.path.exists(output_graph_temp):
        os.remove(output_graph_temp)


if __name__ == '__main__':
    input_checkpoint = '/home/data/hou/workspaces/iobjectspy11/data/out/log/ckpt_model_path/save_model.ckpt'
    net = 'res101'
    output_graph = '/home/data/hou/workspaces/iobjectspy11/data/out/model'
    num_category = 1
    freeze_model(input_checkpoint, net, output_graph, num_category)
