import glob
import os
import shutil

import cv2
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


# 统计水体、道路中的连通块数目
def statis_road_water(mask_path, out_data_path, num_connected_components_w, num_connected_components_r, connectivity,
                      ext):
    images = glob.glob(os.path.join(mask_path, '*.' + ext))

    water_connect_array = np.array([])
    road_connect_array = np.array([])
    water_road_connect_array = np.array([])

    # 输出路径
    output_data_path_visual_wr = os.path.join(out_data_path, 'visual_wr')
    output_data_path_label_w = os.path.join(out_data_path, 'water')
    output_data_path_label_r = os.path.join(out_data_path, 'road')
    # 将连通域处理过后的水体图像转为8位
    output_data_path_label_w_16 = os.path.join(output_data_path_label_w, '16')
    output_data_path_label_w_8 = os.path.join(output_data_path_label_w, '8')
    # 将连通域处理过后的水体图像转为8位
    output_data_path_label_r_16 = os.path.join(output_data_path_label_r, '16')
    output_data_path_label_r_8 = os.path.join(output_data_path_label_r, '8')
    # 创建不存在的输出路径
    if not os.path.exists(output_data_path_visual_wr):
        os.makedirs(output_data_path_visual_wr)
    if not os.path.exists(output_data_path_label_w):
        os.makedirs(output_data_path_label_w)
    if not os.path.exists(output_data_path_label_r):
        os.makedirs(output_data_path_label_r)

    if not os.path.exists(output_data_path_label_w_16):
        os.makedirs(output_data_path_label_w_16)
    if not os.path.exists(output_data_path_label_w_8):
        os.makedirs(output_data_path_label_w_8)

    if not os.path.exists(output_data_path_label_r_16):
        os.makedirs(output_data_path_label_r_16)
    if not os.path.exists(output_data_path_label_r_8):
        os.makedirs(output_data_path_label_r_8)

    for im_f in tqdm(images):
        mask = np.array(Image.open(im_f))
        file_name = os.path.splitext(os.path.basename(im_f))[0]
        num_labels_wr = 0
        num_labels_r_ = 0
        num_labels_w_ = 0
        labels_w_ = np.array([])
        labels_r_ = np.array([])

        # 先统计含有水体的块数量分布
        if np.sum(mask == 3) >= num_connected_components_w:
            water = (mask == 3).astype(np.uint8)  # 01二值化
            num_labels_w, labels_w, stats_w, centers_w = cv2.connectedComponentsWithStats(water,
                                                                                          connectivity=8,
                                                                                          ltype=cv2.CV_16U)
            water_connect_array = np.append(water_connect_array, num_labels_w)
            num_labels_w_ += num_labels_w
            num_labels_wr += num_labels_w
            labels_w_ = labels_w

        # 再统计含有道路的块数量分布
        if np.sum(mask == 4) >= num_connected_components_r:
            road = (mask == 4).astype(np.uint8)  # 01二值化
            num_labels_r, labels_r, stats_r, centers_r = cv2.connectedComponentsWithStats(road,
                                                                                          connectivity=8,
                                                                                          ltype=cv2.CV_16U)
            road_connect_array = np.append(road_connect_array, num_labels_r)
            num_labels_r_ += num_labels_r
            num_labels_wr += num_labels_r
            labels_r_ = labels_r

        water_road_connect_array = np.append(water_road_connect_array, num_labels_wr)

        # 可视化同时满足水体连接域大于num_connected_components_w，道路连接域大于num_connected_components_r的情况
        if (num_labels_r_ > num_connected_components_w) | (num_labels_w_ > num_connected_components_r):
            shutil.copyfile(im_f, os.path.join(output_data_path_visual_wr,
                                               file_name + "." + ext))

        if num_labels_w_ > num_connected_components_w:
            cv2.imwrite(os.path.join(output_data_path_label_w_16,
                                     file_name + "." + ext), labels_w_)
            u16_to_u8(os.path.join(output_data_path_label_w_16,
                                   file_name + "." + ext), os.path.join(output_data_path_label_w_8,
                                                                        file_name + "." + ext))

        if num_labels_r_ > num_connected_components_r:
            cv2.imwrite(os.path.join(output_data_path_label_r_16,
                                     file_name + "." + ext), labels_r_)
            u16_to_u8(os.path.join(output_data_path_label_r_16,
                                   file_name + "." + ext), os.path.join(output_data_path_label_r_8,
                                                                        file_name + "." + ext))

    # time.sleep(0.1)
    num, count = np.unique(water_road_connect_array, return_counts=True)
    print("image count:{}".format(water_road_connect_array.size))
    print(dict(zip(num, count)))

    num, count = np.unique(water_connect_array, return_counts=True)
    print("water count:{}".format(water_connect_array.size))
    print(dict(zip(num, count)))

    num, count = np.unique(road_connect_array, return_counts=True)
    print("road count:{}".format(road_connect_array.size))
    print(dict(zip(num, count)))


def stretch_n(bands, lower_percent=1, higher_percent=99, tile_h=15000, tile_w=15000):
    """
    16 to 8
    :param bands:
    :param lower_percent:
    :param higher_percent:
    :return:
    """
    is_transpose = False
    if np.argmin(bands.shape) == 0:
        is_transpose = True
    if is_transpose:
        bands = np.transpose(bands, (1, 2, 0))
    out = np.zeros_like(bands, dtype=np.uint8)
    h, w, n = bands.shape
    # tile_h,tile_w=15000,15000
    index_h, index_w = h // tile_h + 1, w // tile_w + 1
    total_task = index_h * index_w * n
    task_index = 0
    # print('processing {}/{}'.format(task_index, total_task))
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        for j in range(index_h):
            for k in range(index_w):
                tmp_h, tmp_w = min((j + 1) * tile_h, h), min((k + 1) * tile_w, w)
                t = a + (bands[j * tile_h:tmp_h, k * tile_w:tmp_w, i] - c) * (b - a) / (d - c)
                t[t < a] = a
                t[t > b] = b
                out[j * tile_h:tmp_h, k * tile_w:tmp_w, i] = t.astype(np.uint8)
                task_index += 1
                # print('processing {}/{}'.format(task_index, total_task))

        # t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        # t[t < a] = a
        # t[t > b] = b
        # out[:, :, i] = t.astype(np.uint8)
    if is_transpose:
        out = np.transpose(out, (2, 0, 1))
    return out


# 16位图像转8位图像
def u16_to_u8(in_image, out_image):
    in_f = rasterio.open(in_image)
    prof = in_f.profile
    prof.update({'dtype': rasterio.uint8})
    with rasterio.open(out_image, 'w', **prof) as wf:
        wf.write(stretch_n(in_f.read()))


def from_mask_data_get_origin_data(input_mask_path, input_origin_path, output_origin_path):
    mask_ids = os.listdir(input_mask_path)

    for mask_id in mask_ids:
        if mask_id.endswith(".png"):
            fileid = mask_id.split(".")[0]
            # mask_file = os.path.join(input_mask_path, fileid + ".png")
            origin_file = os.path.join(input_origin_path, fileid + ".jpg")
            shutil.copyfile(origin_file, os.path.join(output_origin_path,
                                                      fileid + ".png"))


def image_image_stitching(input_mask_path, input_origin_path, out_data_path):
    mask_ids = os.listdir(input_mask_path)

    for mask_id in mask_ids:
        im_list = []
        if mask_id.endswith(".png"):
            fileid = mask_id.split(".")[0]
            mask_file = os.path.join(input_mask_path, fileid + ".png")
            origin_file = os.path.join(input_origin_path, fileid + ".jpg")
            im_list.append(Image.open(mask_file).convert("RGB"))
            im_list.append(Image.open(origin_file))
            width, height = im_list[0].size
            result = Image.new(im_list[0].mode, (width, height * len(im_list)))
            for i, im in enumerate(im_list):
                result.paste(im, box=(0, i * height))

            result.save(os.path.join(out_data_path,
                                     fileid + ".png"))


if __name__ == '__main__':
    # 可视化分析预测结果中道路或者水体连通域大于1的情况,并保存8
    input_data_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/input_data/results/results'
    input_origin_path = '/home/hou/Desktop/windowdata/temp/connected_component/img_testA'
    # 每幅影像中水体或者道路的个数
    num_connected_components_w = 1
    num_connected_components_r = 1
    # 连通域：4-邻域；8-邻域
    connectivity = 8

    # out_dir and create_out_dir
    output_data_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp'
    output_origin_path = os.path.join(output_data_path, 'origin_wr')

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    if not os.path.exists(output_origin_path):
        os.makedirs(output_origin_path)

    # 可视化分析预测结果中道路或者水体连通域大于1的情况,并单独提取出水体以及道路分别进行分析。
    # statis_road_water(
    #     input_data_path, output_data_path, num_connected_components_w, num_connected_components_r, connectivity,
    #     "png")
    input_mask_path = os.path.join(output_data_path, 'visual_wr')

    # 从待分析的mask数据中获取原始影像
    # from_mask_data_get_origin_data(input_mask_path, input_origin_path, output_origin_path)

    # 将原始影像与mask进行拼接
    out_image_stitching_wr = os.path.join(output_data_path, "image_stitching_wr")
    if not os.path.exists(out_image_stitching_wr):
        os.makedirs(out_image_stitching_wr)
    image_image_stitching(input_mask_path, input_origin_path, out_image_stitching_wr)

    # 将水体mask与原始影像进行拼接
    out_image_stitching_w = os.path.join(output_data_path, "image_stitching_w")
    if not os.path.exists(out_image_stitching_w):
        os.makedirs(out_image_stitching_w)
    input_mask_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp/water/8'
    image_image_stitching(input_mask_path, input_origin_path, out_image_stitching_w)

    # 将道路mask与原始影像进行拼接
    out_image_stitching_r = os.path.join(output_data_path, "image_stitching_r")
    if not os.path.exists(out_image_stitching_r):
        os.makedirs(out_image_stitching_r)
    input_mask_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp/road/8'
    image_image_stitching(input_mask_path, input_origin_path, out_image_stitching_r)

    # 可视化分析训练数据中道路或者水体大于1的情况
    #
    # input_data_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/input_data/results/results'
    # output_data_path = '/home/data/hou/workspaces/my_knowledge_base/competition/ccf_semantic_segmentation/out/tmp'
    # num_connected_components_w = 1
    # num_connected_components_r = 1
    # connectivity = 8
    # if not os.path.exists(output_data_path):
    #     os.makedirs(output_data_path)
    # statis_road_water(
    #     input_data_path, output_data_path, num_connected_components_w, num_connected_components_r, connectivity,
    #     "png")

    # 可视化分析训练数据中带有道路或者水体的情况
