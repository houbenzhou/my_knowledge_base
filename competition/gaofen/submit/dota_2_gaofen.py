
import argparse
import os
import shutil
import xml.etree.ElementTree as ET


def _save_gaofen(output_path_label, list_onefile_boxs,  image_name):
    """
      生成xml描述文件

      :param output_path_label: 输入标签文件存储路径
      :type output_path_label: str
      :param lists: 包含bbox，category，difficult信息
      :type lists: list
      :param width: 图像宽度
      :type width: Long
      :param height: 图像高度
      :type height: Long
      :param pic_name: 对应标签文件的图片名称
      :type pic_name: str
      :param tile_format: 切片的图像格式:TIFF,PNG,JPG
      :type tile_format: str

      """

    from lxml.etree import Element, SubElement, tostring

    node_root = Element('annotation')

    node_source = SubElement(node_root, 'source')

    node_filename = SubElement(node_source, 'filename')
    node_filename.text = image_name

    node_origin = SubElement(node_source, 'origin')
    node_origin.text = 'GF2/GF3'

    node_research = SubElement(node_root, 'research')
    node_version = SubElement(node_research, 'version')
    node_version.text = str(1.0)

    node_provider = SubElement(node_research, 'provider')
    node_provider.text = 'Supermap'

    node_author = SubElement(node_research, 'author')
    node_author.text = "SuperAI"

    node_pluginname = SubElement(node_research, 'pluginname')
    node_pluginname.text = "FAIR1M"

    node_pluginclass = SubElement(node_research, 'pluginclass')
    node_pluginclass.text = 'object detection'

    node_time = SubElement(node_research, 'time')
    node_time.text = "2021-10"

    node_objects = SubElement(node_root, 'objects')

    for list_onefile_box in list_onefile_boxs:
        node_object = SubElement(node_objects, 'object')

        node_coordinate = SubElement(node_object, 'coordinate')
        node_coordinate.text = "pixel"

        node_type = SubElement(node_object, 'type')
        node_type.text = "rectangle"

        node_description = SubElement(node_object, 'description')
        node_description.text = "None"

        node_possibleresult = SubElement(node_object, 'possibleresult')
        node_name = SubElement(node_possibleresult, 'name')
        node_name.text = str(list_onefile_box[10])
        node_probability = SubElement(node_possibleresult, 'probability')
        node_probability.text = '%s' % list_onefile_box[1]

        node_points = SubElement(node_object, 'points')
        point1 = SubElement(node_points, 'point')
        point1.text = str(str(list_onefile_box[2]) + "," + str(list_onefile_box[3]))
        point2 = SubElement(node_points, 'point')
        point2.text = str(str(list_onefile_box[4]) + "," + str(list_onefile_box[5]))
        point3 = SubElement(node_points, 'point')
        point3.text = str(str(list_onefile_box[6]) + "," + str(list_onefile_box[7]))
        point4 = SubElement(node_points, 'point')
        point4.text = str(str(list_onefile_box[8]) + "," + str(list_onefile_box[9]))
        point5 = SubElement(node_points, 'point')
        point5.text = str(str(list_onefile_box[2]) + "," + str(list_onefile_box[3]))

    xml = tostring(node_root, pretty_print=True, encoding='UTF-8')
    # save_xml = os.path.join(output_path_label, pic_name.split('.')[0] + '.xml')

    with open(output_path_label, 'wb') as f:
        f.write(xml)

## 评估图片目标的长宽比
from PIL import Image, ImageDraw
def dota_2_gaofen(input_image,input_dota_data,out_gaofen_data,score_thresh):
    image_names = os.listdir(input_image)
    task_category_names = os.listdir(input_dota_data)
    for image_name in image_names:
        file_name=image_name.split(".png")[0]
        list_onefile_boxs=[]
        for task_category_name in task_category_names:
            category_name=task_category_name.split("_")[1].split('.txt')[0]
            task_category_file=os.path.join(input_dota_data,task_category_name)
            task_category_file_temp = open(task_category_file, "r", encoding="utf-8", errors="ignore")
            try:
                while True:
                    mystr = task_category_file_temp.readline()  # 表示一次读取一行
                    if not mystr:
                        # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                        break
                    mystr = mystr.rstrip("\n")
                    list_bbox=mystr.split(' ')
                    if list_bbox[0]==file_name:
                        if float(list_bbox[1])>=score_thresh:
                            list_bbox.append(category_name)
                            list_onefile_boxs.append(list_bbox)

            except IOError:
                print(IOError)
        output_path_label=os.path.join(out_gaofen_data,file_name+'.xml')
        _save_gaofen(output_path_label, list_onefile_boxs,  image_name)


if __name__ == '__main__':
    input_image=r"C:\Users\houbenzhou\Desktop\result_obb\images"
    input_dota_data=r"C:\Users\houbenzhou\Desktop\result_obb\Task1_results_nms"
    out_gaofen_data=r"C:\Users\houbenzhou\Desktop\result_obb\gaofen_Task1_results_nms"
    score_thresh=0.5
    if not os.path.exists(out_gaofen_data):
        os.makedirs(out_gaofen_data)
    dota_2_gaofen(input_image,input_dota_data,out_gaofen_data,score_thresh)


