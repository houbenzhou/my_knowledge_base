import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
def count_mean_var( filepath, tile_size):
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

    num = len(pathDir) * tile_size * tile_size  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename))
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    image_mean = []
    image_mean.append(float(R_mean))
    image_mean.append(float(G_mean))
    image_mean.append(float(B_mean))
    image_std = []
    image_std.append(float(R_var))
    image_std.append(float(G_var))
    image_std.append(float(B_var))

    # print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    # print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
    return image_mean, image_std

if __name__ == '__main__':
    filepath = 'E://workspaces//data//0_object_detection_data//0_dota//0_dota_v2_800//VOC//Images'
    tile_size= 800
    image_mean, image_std = count_mean_var(filepath, tile_size)
    print(image_mean)
    print(image_std)