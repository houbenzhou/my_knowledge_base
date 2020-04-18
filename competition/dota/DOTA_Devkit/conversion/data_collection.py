import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def voc_data_collection(first_voc_path, second_voc_path, target_voc_path, prefix):
    first_images_path = os.path.join(first_voc_path, "Images")
    first_labels_path = os.path.join(first_voc_path, "Annotations")
    second_images_path = os.path.join(second_voc_path, "Images")
    second_labels_path = os.path.join(second_voc_path, "Annotations")
    target_images_path = os.path.join(target_voc_path, "Images")
    target_labels_path = os.path.join(target_voc_path, "Annotations")


    copy_allfiles(first_images_path, target_images_path)
    copy_allfiles(first_labels_path, target_labels_path)
    image_names = os.listdir(target_images_path)
    label_names = os.listdir(target_labels_path)
    for label_name in label_names:
        origin_label_path = os.path.join(target_labels_path, label_name)
        new_label_path = os.path.join(target_labels_path, prefix + label_name)
        os.rename(origin_label_path, new_label_path)
    for image_name in image_names:
        origin_image_path = os.path.join(target_images_path, image_name)
        new_image_path = os.path.join(target_images_path, prefix + image_name)
        os.rename(origin_image_path, new_image_path)
    copy_allfiles(second_images_path, target_images_path)
    copy_allfiles(second_labels_path, target_labels_path)



import os
import shutil

def copy_allfiles(src,dest):
#src:原文件夹；dest:目标文件夹
  src_files = os.listdir(src)
  for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest)

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
def copy_sda_files(source_sda_path,target_path):
    if os.path.isfile(source_sda_path):
        shutil.copy(source_sda_path, target_path)

if __name__ == '__main__':
    first_voc_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/out/2020-04-18/1024_collection_2_3and_delete4_5/mountain_train_test_1024_2_3/VOC'
    second_voc_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/out/2020-04-18/1024_collection_2_3and_delete4_5/mountain_train_test_1024_delete_4_5/VOC'
    target_voc_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/out/2020-04-18/1024_collection_2_3and_delete4_5/VOC'
    voc_labels_path = os.path.join(target_voc_path, "Annotations")
    voc_images_path = os.path.join(target_voc_path, "Images")
    voc_main_path = os.path.join(target_voc_path, "ImageSets", "Main")
    source_sda_path = os.path.join(first_voc_path, "VOC.sda")
    if not os.path.exists(voc_labels_path):
        os.makedirs(voc_labels_path)
    if not os.path.exists(voc_images_path):
        os.makedirs(voc_images_path)
    if not os.path.exists(voc_main_path):
        os.makedirs(voc_main_path)

    #voc数据集融合，仅支持相同尺度的图像
    voc_data_collection(first_voc_path, second_voc_path, target_voc_path, prefix='0')
    #  生成VOC的索引文件
    _save_index_file(voc_main_path, voc_labels_path)
    # 拷贝sda文件
    copy_sda_files(source_sda_path,target_voc_path)
