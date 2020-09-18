import numpy as np
import rasterio


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
        wf.write(stretch_n(in_f.read(),0.1,99.9))


if __name__ == '__main__':
    # u16_to_u8()
    input_path = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/test/images/1.tiff'
    out_path = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/test/images1/1.tif'
    u16_to_u8(input_path, out_path)
