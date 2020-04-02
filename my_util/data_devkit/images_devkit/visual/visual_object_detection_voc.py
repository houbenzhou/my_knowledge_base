import os
import shutil
import xml.etree.ElementTree as ET


from PIL import Image, ImageDraw


def eval_objects_width_height(xml_paths, object_width_height_txt):
    """
    评估目标的长宽比是否正常
    :param xml_paths:
    :param object_width_height_txt:
    :return:
    """
    xml_path_ = os.listdir(xml_paths)
    for pic_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_path = os.path.join(xml_paths, pic_name)
        tree = ET.parse(xml_path)
        rect = {}
        root = tree.getroot()
        for name in root.iter('path'):
            rect['path'] = name.text
        for ob in root.iter('object'):
            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = xmin.text
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = ymin.text
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = xmax.text
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = ymax.text
            line = "图片:" + "y/x:" + str(
                (float(rect['ymax']) - float(rect['ymin'])) / (float(rect['xmax']) - float(rect['xmin'])))
            object_width_height_txt.write(line + '\n')


def visual_object_detection_voc(voc_path, out_path):
    """
    可视化voc数据集
    :param voc_path:
    :param out_path:
    :return:
    """
    voc_xml = os.path.join(voc_path, 'Annotations')
    voc_img = os.path.join(voc_path, 'Images')
    xml_path_ = os.listdir(voc_xml)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_pth = os.path.join(voc_xml, xml_name)
        img_name = xml_name.split(".")[0] + ".jpg"
        img_pth = os.path.join(voc_img, img_name)

        img = Image.open(img_pth)
        draw = ImageDraw.Draw(img)
        tree = ET.parse(xml_pth)
        rect = {}
        root = tree.getroot()
        for ob in root.iter('object'):
            for name in ob.iter('name'):
                rect['name'] = name.text
            for bndbox in ob.iter('bndbox'):
                for xmin in bndbox.iter('xmin'):
                    rect['xmin'] = float(xmin.text)
                for ymin in bndbox.iter('ymin'):
                    rect['ymin'] = float(ymin.text)
                for xmax in bndbox.iter('xmax'):
                    rect['xmax'] = float(xmax.text)
                for ymax in bndbox.iter('ymax'):
                    rect['ymax'] = float(ymax.text)
                draw.line((rect['xmin'], rect['ymin'], rect['xmax'], rect['ymin']), fill=(255, 0, 0), width=4)
                draw.line((rect['xmax'], rect['ymin'], rect['xmax'], rect['ymax']), fill=(255, 0, 0), width=4)
                draw.line((rect['xmax'], rect['ymax'], rect['xmin'], rect['ymax']), fill=(255, 0, 0), width=4)
                draw.line((rect['xmin'], rect['ymax'], rect['xmin'], rect['ymin']), fill=(255, 0, 0), width=4)
                draw.text((rect['xmin'], rect['ymin'] - 44), rect['name'], fill="#0000ff")
        img.save(os.path.join(out_path, img_name))


if __name__ == '__main__':
    # plane_1024
    # voc_path = '/home/data/windowdata/data/rssrai2019_hou/1024/VOC'
    # out_path = '/home/data/windowdata/temp/visual2'
    # large-vehicle 1024
    # voc_path = '/home/data/windowdata/data/rssrai2019_hou/1024/large-vehicle/VOC'
    # out_path = '/home/data/windowdata/temp/visual3'
    # plane 1024
    voc_path = '/home/data/hou/workspaces/iobjectspy/resources_ml/out/iobjectspy_voc'
    out_path = '/home/data/windowdata/temp/visual'
    # large-vehicle 1024
    # voc_path = '/home/data/windowdata/data/rssrai2019_hou/600/large-vehicle/VOC'
    # out_path = '/home/data/windowdata/temp/visual5'
    visual_object_detection_voc(voc_path, out_path)