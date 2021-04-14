import numpy as np
import rasterio
import os


def stretch_n(bands, lower_percent=2, higher_percent=98):
    """
    将影像数据标准化到0到1之间


    :param bands:  输入影像
    :param lower_percent:   最小值比率
    :param higher_percent:  最大值比率
    :return:  标准化后的影像
    """
    is_transpose = False
    if np.argmin(bands.shape) == 0:
        is_transpose = True
    if is_transpose:
        bands = np.transpose(bands, (1, 2, 0))
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    if is_transpose:
        out = np.transpose(out, (2, 0, 1))
    return out


def u16_to_u8(in_image, out_image):
    in_f = rasterio.open(in_image)
    prof = in_f.profile
    prof.update({'dtype': rasterio.uint8})
    with rasterio.open(out_image, 'w', **prof) as wf:
        wf.write(stretch_n(in_f.read(), 0.1, 99.9))


if __name__ == '__main__':
    curr_dir = os.path.join('E:\\supermap', '2_ai_example', 'demov3(20210310)')

    InPath = os.path.join(curr_dir, 'data', '2_test', '16')
    OutFile = os.path.join(curr_dir, 'data', '2_test', '8', 'demo_8.tiff')
    # input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\16\demo_16.tiff'
    # out_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\8_percent\demo_8.tiff'
    input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\1.tif'
    out_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\1_8.tif'

    u16_to_u8(input_path, out_path)