import argparse
import os
import re
from collections import OrderedDict

import yaml
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_xml(image_name, lists, save_dir, width=1609, height=500, channel=3):
    '''
  :param image_name:图片名
  :param bbox:对应的bbox
  :param save_dir:
  :return:
  '''
    from lxml.etree import Element, SubElement, tostring
    from xml.dom.minidom import parseString

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')

    node_filename.text = str(image_name.replace('png', 'jpg'))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    # segmented
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '%s' % 0
    # lists=[[1,2,3,4,'cat'],[2,3,4,5,'car'],[5,6,9,8,'test']]
    for list in lists:
        xmin = list[0]
        ymin = list[1]
        xmax = list[2]
        ymax = list[3]
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = list[4]
        pose = SubElement(node_object, 'pose')
        pose.text = 'Unspecified'
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = str(list[5])

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % xmin
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % ymin
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % xmax
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % ymax
        # if (xmax > xmin) | (ymax > ymin):
        #     print(listnew)
        #     print(pic_name)
    del lists[:]
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('png', 'xml'))

    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


def save_config_to_yaml(config: OrderedDict, yaml_file: str, encoding='utf8') -> None:
    """
    save the config to a yaml format file
    :param config:
    :param yaml_file:
    :param encoding:
    :return:
    """
    with open(yaml_file, 'w', encoding=encoding) as f:
        ordered_yaml_dump(config, f, encoding='utf8', allow_unicode=True)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def create_annotation(path_images, path_label, categorys, tile_size, tile_offset, target_label_path, sda_path):
    pic_names = os.listdir(path_images)
    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)
    dict_categorys = set([])
    categorys_temp = ['__background__']
    for pic_name in pic_names:
        # 获取图片路径，用于获取图像大小以及通道数
        images_pth = os.path.join(path_images, pic_name)
        img = Image.open(images_pth)
        # 测试图像为RGB
        # print(img.mode)
        # 获取图像尺寸
        width, height = img.size
        channel = 3
        # 获取label路径，用于获取bbox以及类名

        label_name = pic_name.split('.')
        if label_name[-1] == "png":
            label_name[-1] = 'txt'
            label_name = str.join(".", label_name)
        lists = []
        label_path = os.path.join(path_label, label_name)
        file = open(label_path, "r", encoding="utf-8", errors="ignore")
        try:
            while True:
                mystr = file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                list = mystr.split(' ')
                if len(list)>=8:
                    x = []
                    y = []
                    x.append(float(list[0]))
                    x.append(float(list[2]))
                    x.append(float(list[4]))
                    x.append(float(list[6]))
                    y.append(float(list[1]))
                    y.append(float(list[3]))
                    y.append(float(list[5]))
                    y.append(float(list[7]))
                    xmin = min(x)
                    ymin = min(y)
                    xmax = max(x)
                    ymax = max(y)
                    name = list[8]
                    dict_categorys.add(name)
                    difficult = int(list[9][0])

                    if categorys is None:
                        if (ymax > (ymin + 2)) & (xmax > (xmin + 2)) & (difficult <= 2):
                            if ((ymax - ymin) / (xmax - xmin) <= 9) & ((ymax - ymin) / (xmax - xmin) >= 0.13):
                                listnew = []
                                listnew.append(xmin)
                                listnew.append(ymin)
                                listnew.append(xmax)
                                listnew.append(ymax)

                                listnew.append(name)
                                if difficult <= 1:
                                    listnew.append(difficult)
                                else:
                                    listnew.append(1)

                                lists.append(listnew)

                                category = name
                                # 获取数据集中category字段保存的类别
                                if category not in categorys_temp:
                                    categorys_temp.append(category)

                    else:
                        if name in categorys:
                            if (ymax > (ymin + 2)) & (xmax > (xmin + 2)) & (difficult <= 2):
                                if ((ymax - ymin) / (xmax - xmin) <= 9) & ((ymax - ymin) / (xmax - xmin) >= 0.13):
                                    listnew = []
                                    listnew.append(xmin)
                                    listnew.append(ymin)
                                    listnew.append(xmax)
                                    listnew.append(ymax)

                                    listnew.append(name)
                                    if difficult <= 1:
                                        listnew.append(difficult)
                                    else:
                                        listnew.append(1)
                                    lists.append(listnew)
        except IOError:
            print(IOError)
        if lists:
            save_xml(pic_name, lists, target_label_path, width, height,
                     channel)

        # 创建VOC数据描述配置文件
    if categorys is not None:
        for category in categorys:
            categorys_temp.append(category)
    # 1024 更新tile记录数
    dic_voc_yml = OrderedDict({
        'dataset': OrderedDict({"name": "example_voc",
                                'classes': categorys_temp,
                                'image_count': 0,
                                "data_type": "voc",
                                "input_bandnum": 3, "input_ext": 'tif',
                                "x_ext": 'jpg',
                                "tile_size_x": tile_size,
                                "tile_size_y": tile_size,
                                "tile_offset_x": tile_offset,
                                "tile_offset_y": tile_offset,
                                "image_mean": [115.6545965, 117.62014299, 106.01483799],
                                "image_std": [56.82521775, 53.46318049, 56.07113724]}),

    })
    save_config_to_yaml(dic_voc_yml, sda_path)


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
        train_txt.write(label_name + os.linesep)
    for pic_name in list_val:
        label_name = pic_name.split('.')[0]
        val_txt.write(label_name + os.linesep)
    for pic_name in list_test:
        label_name = pic_name.split('.')[0]
        test_txt.write(label_name + os.linesep)
    for pic_name in list_trainval:
        label_name = pic_name.split('.')[0]
        trainval_txt.write(label_name + os.linesep)

    # 关闭所有打开的文件
    train_txt.close()
    val_txt.close()
    test_txt.close()
    trainval_txt.close()


def create_images(path, voc_labels_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    label_names = os.listdir(voc_labels_path)

    for label_name in label_names:

        pic_name = label_name.split('.')
        if pic_name[-1] == "xml":
            pic_name[-1] = 'png'
            pic_name = str.join(".", pic_name)
        voc_pic_name = label_name.split('.')
        if voc_pic_name[-1] == "xml":
            voc_pic_name[-1] = 'jpg'
            voc_pic_name = str.join(".", voc_pic_name)
        im = Image.open(os.path.join(path, pic_name))
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im)

        bg.save(out_path + "/" + voc_pic_name)


def get_parser():
    parser = argparse.ArgumentParser(description="split dota")
    parser.add_argument(
        "--input_dota_path",
        default="/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/airplane/dota_format/val",
        help="Base path for dota data",
    )

    parser.add_argument(
        "--out_voc_path",
        default='/home/data/hou/experiments/gaofen4/data/test_data/val/voc',
        help="Output base path for dota data",
    )
    parser.add_argument(
        "--category",
        # default='plane,storage-tank,tennis-court,ground-track-field,large-vehicle',
        default=None,
    help = "Output base path for dota data",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=800,
        help="overrlap between two patches ",
    )
    parser.add_argument(
        "--tile_offset",
        type=int,
        default=400,
        help="subsize of patch ",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    input_dota_path = args.input_dota_path
    out_voc_path = args.out_voc_path
    category = args.category
    tile_size = args.tile_size
    tile_offset = args.tile_offset
    path_images = os.path.join(input_dota_path, "images")
    path_label = os.path.join(input_dota_path, "labelTxt")
    voc_labels_path = os.path.join(out_voc_path, "Annotations")
    voc_images_path = os.path.join(out_voc_path, "Images")
    voc_main_path = os.path.join(out_voc_path, "ImageSets", "Main")
    sda_path = os.path.join(out_voc_path, "VOC.sda")
    # 将类别的字符串按照,号或者，号进行分割
    if category is not None:
        regex = ",|，"
        category = re.split(regex, category)
    # 生成VOC的标签数据
    create_annotation(path_images, path_label, category, tile_size, tile_offset, voc_labels_path, sda_path)
    # 生成VOC的图像数据
    create_images(path_images, voc_labels_path, voc_images_path)
    # # 生成VOC的索引文件
    _save_index_file(voc_main_path, voc_labels_path)
