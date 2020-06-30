# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
# -*-coding:utf-8-*-
# pip install mlflow

import mlflow as ml

# mlflow记录试验参数分为两步：
# 1.记录训练的超参数，调用log_param()
# 2.记录训练出的模型验证过程得出的指标，调用log_metric()

# 参数以 lr,mini_mask_shape,epoch 为例
lr = 0.001
mini_mask_shape = [28, 28]
epoch = 20
# mlrun保存日志路径
ml.set_tracking_uri('file:///home/data/windowdata/mlruns')
# 通过设置不同的实验id来管理实验,建议这一层级为项目名称，比如：iobjectspy_faster_rcnn_dota
experiment_id = 'mlflow_experiment_id'
ml.set_experiment(experiment_id)
# 通过设置
ml.set_tag('experiment_id', 'v1')
# log_param的字符参数可以随意设置，代表了记录的参数的值（如lr)在记录结果中对应的名称
ml.log_param('lr', lr)
ml.log_param('mini_mask_shape', mini_mask_shape)
ml.log_param('epoch', epoch)

# 如果想以命名区分每次试验，可以将experiment_name作为参数记录
experiment_name = 'binary_test'
ml.log_param('name', experiment_name)

# 指标以val_loss为例
val_loss = 1

# 至此记录训练中参数和指标的工作已经完成，而因为指标是不断变化的，所以需要不断更新其值。
# 以下循环代替训练中的迭代，每个epoch结束后val_loss值更新，并被记录
# 代码运行完后的操作见README.md
for i in range(epoch):
    val_loss = val_loss - 0.01
    ml.log_metric('val_loss', val_loss)
