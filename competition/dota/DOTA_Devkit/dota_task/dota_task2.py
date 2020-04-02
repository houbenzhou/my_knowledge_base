import os
import shutil
from typing import re

import yaml
from dotmap import DotMap


def _create_dota_task2(input_data, out_file, categoty):
    with open(out_file, 'a') as file_out:
        label_names = os.listdir(input_data)
        for label_name in label_names:
            label_file = os.path.join(input_data, label_name)
            file = open(label_file, "r", encoding="utf-8", errors="ignore")

            while True:
                mystr = file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                label_file_list = mystr.split(' ')
                pic_name = label_file_list[1]
                score = label_file_list[2]
                xmin = label_file_list[3]
                ymin = label_file_list[4]
                xmax = label_file_list[5]
                ymax = label_file_list[6]
                categoty_from_list = str(label_file_list[0])
                if categoty_from_list == categoty:
                    outline = pic_name + ' ' + str(score) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax
                    file_out.write(outline )


def create_dota_task2(input_data, out_path, categotys):
    """
    可视化voc数据集
    :param voc_path:
    :param out_path:
    :return:
    """
    try:
        with open(categotys) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config = DotMap(config_dict)
        # 模型中支持的所有类别
        config.get("model").get("categorys").remove("__background__")
        category_name = config.get("model").get("categorys")
        category_name = [str(i) for i in category_name]
        categotys = category_name
    except:
        regex = ",|，"
        categotys = re.split(regex, categotys)
    for categoty in categotys:
        out_name = "Task2_" + categoty + '.txt'
        out_file = os.path.join(out_path, out_name)
        _create_dota_task2(input_data, out_file, categoty)


if __name__ == '__main__':
    input_data = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/competition/dotav2/test/images/labelTxt'
    out_path = '/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/out'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    categotys = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/competition/saved_model_10/saved_model_10.sdm'
    create_dota_task2(input_data, out_path, categotys)
