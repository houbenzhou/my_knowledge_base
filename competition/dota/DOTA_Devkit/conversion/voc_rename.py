import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rename_voc_data(path_images, path_label, prefix):
    label_names = os.listdir(path_label)
    image_names = os.listdir(path_images)

    for label_name in label_names:
        origin_label_path = os.path.join(path_label, label_name)
        new_label_path = os.path.join(path_label, prefix + label_name)
        os.rename(origin_label_path, new_label_path)
    for image_name in image_names:
        origin_image_path = os.path.join(path_images, image_name)
        new_image_path = os.path.join(path_images, prefix + image_name)
        os.rename(origin_image_path, new_image_path)


def _save_index_file(output_path_main, output_path_img):
    if not os.path.exists(output_path_main):
        os.makedirs(output_path_main)
    # 随机将数据分为train、val、test数据
    pic_names = os.listdir(output_path_img)
    # 分配训练数据验证数据的数组长度
    train_length = int((len(pic_names) / 5) * 4)
    val_length = int(len(pic_names) / 10)
    # 训练数据集、验证数据集、测试数据集数组
    list_train = pic_names[0:train_length]
    list_val = pic_names[train_length:train_length + val_length]
    list_test = pic_names[train_length:]
    list_trainval = list_train + list_val
    # 打开创建的文件
    train_txt = open(os.path.join(output_path_main, 'train.txt'), "w")
    val_txt = open(os.path.join(output_path_main, 'val.txt'), "w")
    test_txt = open(os.path.join(output_path_main, 'test.txt'), "w")
    trainval_txt = open(os.path.join(output_path_main, 'trainval.txt'), "w")

    for pic_name in list_train:
        label_name = pic_name.split('.')[0]
        train_txt.write(label_name + '\n')
    for pic_name in list_val:
        label_name = pic_name.split('.')[0]
        val_txt.write(label_name + '\n')
    for pic_name in list_test:
        label_name = pic_name.split('.')[0]
        test_txt.write(label_name + '\n')
    for pic_name in list_trainval:
        label_name = pic_name.split('.')[0]
        trainval_txt.write(label_name + '\n')

    # 关闭所有打开的文件
    train_txt.close()
    val_txt.close()
    test_txt.close()
    trainval_txt.close()


if __name__ == '__main__':
    voc_path = '/home/data/temp/VOC'
    voc_labels_path = os.path.join(voc_path, "Annotations")
    voc_images_path = os.path.join(voc_path, "Images")
    voc_main_path = os.path.join(voc_path, "ImageSets", "Main")
    sda_path = os.path.join(voc_path, "VOC.sda")
    prefix = '0'
    # 生成VOC的标签数据
    rename_voc_data(voc_images_path, voc_labels_path, prefix)
    # # 生成VOC的索引文件
    _save_index_file(voc_main_path, voc_labels_path)
