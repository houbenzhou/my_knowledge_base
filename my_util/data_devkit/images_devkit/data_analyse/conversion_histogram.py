import numpy as np
import rasterio
import os
from matplotlib import pyplot as plt
eplison = 1e-12

def conversion_histogram_16(in_image, lower_percent=2, higher_percent=98):
    bands = rasterio.open(in_image).read()
    is_transpose = False
    if np.argmin(bands.shape) == 0:
        is_transpose = True
    if is_transpose:
        bands = np.transpose(bands, (1, 2, 0))
    n = bands.shape[2]
    # bins_temp = []
    # for j in range(255):
    #     if j==0:
    #         bins_temp.append(0)
    #     elif j==1:
    #         bins_temp.append(256)
    #     else:
    #         bins_temp.append(256 + bins_temp[j - 1])
    for i in range(n):
        bins_temp = []
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        e = (d - c) / 20
        for f in range(20):
            bins_temp.append(c + e * f)
        bands_histogram_temp = bands[:, :, i]
        bands_histogram = bands_histogram_temp.reshape(bands_histogram_temp.shape[0] * bands_histogram_temp.shape[1], )
        bands_histogram = bands_histogram[bands_histogram >= eplison]
        c = np.percentile(bands_histogram, lower_percent)
        d = np.percentile(bands_histogram, higher_percent)

        plt.hist(bands_histogram, bins=bins_temp)
        plt.title("histogram")
        plt.show()
        # hist, bins = np.histogram(bands_histogram, bins=bins_temp)
        # print(hist,bins)

def conversion_histogram_8(in_image, lower_percent=2, higher_percent=98):
    bands = rasterio.open(in_image).read()
    is_transpose = False
    if np.argmin(bands.shape) == 0:
        is_transpose = True
    if is_transpose:
        bands = np.transpose(bands, (1, 2, 0))
    n = bands.shape[2]
    # bins_temp = []
    # for j in range(255):
    #     if j==0:
    #         bins_temp.append(0)
    #     elif j==1:
    #         bins_temp.append(256)
    #     else:
    #         bins_temp.append(256 + bins_temp[j - 1])
    for i in range(n):
        bins_temp = []
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        e = (d - c) / 20
        for f in range(20):
            bins_temp.append(c + e * f)
        bands_histogram_temp = bands[:, :, i]
        bands_histogram = bands_histogram_temp.reshape(bands_histogram_temp.shape[0] * bands_histogram_temp.shape[1], )
        bands_histogram = bands_histogram[bands_histogram >= eplison]
        plt.hist(bands_histogram, bins=bins_temp)
        plt.title("histogram")
        plt.show()
        # hist, bins = np.histogram(bands_histogram, bins=bins_temp)
        # print(hist,bins)



if __name__ == '__main__':
    curr_dir = os.path.join('E:\\supermap', '2_ai_example', 'demov3(20210310)')
    input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\16\demo_16.tiff'
    # input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\8_percent\demo_8.tiff'
    # conversion_histogram_16(input_path,2,98)
    # input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\8_std\demo_8.tiff'
    # input_path = r'E://workspaces//data//0_object_detection_data//0_dota//0_dota_v2_800//VOC//Images//P0022__1__2800___4900.jpg'
    # conversion_histogram_8(input_path)
    input_path = r'E://workspaces//data//0_object_detection_data//0_dota//0_dota_v2_800//VOC//Images'
    image_datas_list = os.listdir(input_path)
    for image_name in image_datas_list:
        images_pth = os.path.join(input_path, image_name)
        conversion_histogram_8(images_pth)

