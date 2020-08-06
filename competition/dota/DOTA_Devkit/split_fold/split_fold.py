import os
import shutil


def copy_allfiles(input_path, target_path, target_path_first_name, target_path_second_name, rate):
    src_files = os.listdir(input_path)
    train_length = int(len(src_files) * rate)
    list_train = src_files[0:train_length]
    list_val = src_files[train_length:]
    for file_name in list_train:
        full_file_name = os.path.join(input_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(target_path, target_path_first_name))
    for file_name in list_val:
        full_file_name = os.path.join(input_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(target_path, target_path_second_name))


def copy_allfiles_from_input(input_path, target_path, path_name):
    path_names = os.listdir(path_name)
    src_files = os.listdir(input_path)

    for file_name in src_files:

        if file_name.split('.')[0] + ".png" in path_names:
            full_file_name = os.path.join(input_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, target_path)


if __name__ == '__main__':
    input_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train_val/images'
    target_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val'
    target_path_first_name = 'train/images'
    target_path_second_name = 'val/images'
    rate = 0.7
    if not os.path.exists(os.path.join(target_path, target_path_first_name)):
        os.makedirs(os.path.join(target_path, target_path_first_name))
    if not os.path.exists(os.path.join(target_path, target_path_second_name)):
        os.makedirs(os.path.join(target_path, target_path_second_name))

    # copy_allfiles(input_path, target_path, target_path_first_name, target_path_second_name, rate)
    input_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train_val/labelTxt'
    path_name = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train/images'
    target_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train/labelTxt'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    copy_allfiles_from_input(input_path, target_path, path_name)
    input_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train_val/labelTxt'
    path_name = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/val/images'
    target_path = '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/val/labelTxt'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    copy_allfiles_from_input(input_path, target_path, path_name)
