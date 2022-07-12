import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_annotation(path_label, dota_labels_path):
    label_names = os.listdir(path_label)
    if not os.path.exists(dota_labels_path):
        os.makedirs(dota_labels_path)

    for label_name in label_names:
        anno_file = os.path.join(path_label, label_name)
        label_name1 = str(label_name.split('.')[0]) + '.txt'

        out_file = os.path.join(dota_labels_path, label_name1)

        with open(out_file, 'w') as file_out:

            tree_objects = ET.parse(anno_file)
            tree = tree_objects.find("objects")

            outline = 'imagesource:GoogleEarth'
            file_out.write(outline + '\n')
            outline = 'gsd:1'
            file_out.write(outline + '\n')
            for obj in tree.findall("object"):
                cls = obj.find("possibleresult").find('name').text
                bboxes = obj.find("points").findall("point")
                list_point = []
                for bbox in bboxes:
                    list_point.append(bbox.text.split(',')[0])
                    list_point.append(bbox.text.split(',')[1])
                # outline = str(list_point[0]) + ' ' + str(list_point[1]) + ' ' + str(list_point[2]) + ' ' + str(
                #     list_point[3]) + ' ' + str(list_point[4]) + ' ' + str(list_point[5]) + ' ' + str(
                #     list_point[6]) + ' ' + str(list_point[7]) + ' ' + cls + ' ' + str(0)
                outline = str(int(float(list_point[0]))) + ' ' + str(int(float(list_point[1]))) + ' ' + str(int(float(list_point[2]))) + ' ' + str(
                    int(float(list_point[3]))) + ' ' + str(int(float(list_point[4]))) + ' ' + str(int(float(list_point[5]))) + ' ' + str(
                    int(float(list_point[6]))) + ' ' + str(int(float(list_point[7]))) + ' ' + cls + ' ' + str(0)
                file_out.write(outline + '\n')




if __name__ == '__main__':


    input_gaofen5_label=r"C:\Users\houbenzhou\Desktop\result_obb\commit_05\gaofen4Task1_results_nms_005_modify"
    output_dota_label=r"C:\Users\houbenzhou\Desktop\result_obb\commit_05\gaofen4Task1_results_nms_005_modify_dota"

    create_annotation(input_gaofen5_label, output_dota_label)


