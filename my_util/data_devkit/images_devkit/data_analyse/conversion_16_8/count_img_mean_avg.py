import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread

# filepath = os.path.join('D:\\workspaces','iobjectspy','resources_ml','example_data','qb_data','out','VOC','Images') # 数据集目录
# filepath = os.path.join('E://workspaces','iobjectspy','resources_ml','example_data','qb_data','out','VOC','Images') # 数据集目录
# filepath = 'E://workspaces//data//0_object_detection_data//0_dota//0_dota_v2_800//VOC//Images'
curr_dir = os.path.join('E:\\supermap', '2_ai_example', 'demov3(20210310)')

# filepath = os.path.join(curr_dir, 'data', '2_test', '16')
filepath = os.path.join(curr_dir, 'data', '2_test', '8')
filepath = r'E:\workspaces\iobjectspy\resources_ml\out\VOC\Images'
pathDir = os.listdir(filepath)
R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename))
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(pathDir) * 512 * 512  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename)) / 255.0
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
