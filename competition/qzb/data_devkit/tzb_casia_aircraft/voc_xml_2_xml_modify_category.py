import json, codecs, os
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import argparse
import os
import shutil
import xml.etree.ElementTree as ET

## 评估图片目标的长宽比
from PIL import Image, ImageDraw
from PIL.ImageFile import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None




def visual_object_detection_voc(input_voc_xml, out_voc_xml,label_dict):
    # voc_xml = os.path.join(voc_path, 'Annotations')
    # voc_img = os.path.join(voc_path, 'Images')
    xml_path_ = os.listdir(input_voc_xml)


    for xml_name in xml_path_:
        # 获取图片路径，用于获取图像大小以及通道数
        xml_pth = os.path.join(input_voc_xml, xml_name)
        tree = ET.parse(xml_pth)
        rect = {}
        root = tree.getroot()
        with codecs.open(os.path.join(out_voc_xml, xml_name ), "w", "utf-8") as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC' + '</folder>\n')
            for filename in root.iter('filename'):
                xml.write('\t<filename>' + filename.text + '</filename>\n')
            xml.write('\t<size>\n')
            for width in root.iter('width'):
                xml.write('\t<width>' + width.text + '</width>\n')
            for height in root.iter('height'):
                xml.write('\t<height>' + height.text + '</height>\n')
            for channels in root.iter('depth'):
                xml.write('\t\t<depth>' + channels.text + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t\t<segmented>0</segmented>\n')
            for ob in root.iter('object'):
                for name in ob.iter('name'):
                    label = name.text
                for bndbox in ob.iter('bndbox'):
                    for xmin in bndbox.iter('xmin'):
                        xmin = float(xmin.text)
                    for ymin in bndbox.iter('ymin'):
                        ymin = float(ymin.text)
                    for xmax in bndbox.iter('xmax'):
                        xmax = float(xmax.text)
                    for ymax in bndbox.iter('ymax'):
                        ymax = float(ymax.text)

                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    label = label_dict[label]
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + label + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>0</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')
                    xml.write('\t\t\t<xmin>' + str(int(round(xmin))) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(int(round(ymin))) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(int(round(xmax))) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(int(round(ymax))) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')

            xml.write('</annotation>')




def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--input_voc_xml",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_voc\Annotations",
        help="voc image path",
    )
    parser.add_argument(
        "--out_voc_xml",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\modify_casia_2_qzb_voc\Annotations",
        help="voc label path",
    )
    parser.add_argument(
        "--label_dict",
        default=r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\origin_voc\New_Annotations",
        help="voc label path",
    )

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    input_voc_xml = args.input_voc_xml
    out_voc_xml = args.out_voc_xml
    label_dict=({ 'F-15': '00','F-2': '00','F-4': '00','F-16': '00','F-A-18': '00','F-A-18E-F': '00','F-22': '00','F-35': '00','SU-27': '00','MIGE-29': '00','MIGE-31': '00','ZHENFENG': '00','TAIFENG': '00','HUANYING2000': '00','F-117': '00','AV-8B': '00','L-159': '00',
                  'A-10': '01','SU-24': '01','SU-25': '01','SU-34': '01',
                  'B-2': '02','B-1B': '02','B-52': '02','TU-22M': '02','TU-22M3': '02','TU-95': '02','TU-160': '02',
                  'E-2': '03','E-3': '03','E-8': '03','E-737': '03',
                  'RC-135S': '04','RC-135V-W': '04','U-28A': '04','WC-135': '04','SHAOBING-R1': '04',
                  'MQ-9': '05',
                  'C-130': '06','C-17': '06','C-5': '06','C-2': '06','C-12': '06','AN-12': '06','YIER-18': '06','YIER-76': '06','C-21': '06','C-32': '06','AN-26': '06',
                  'P-3C': '07','P-8A': '07',
                  'KC-135': '08','KC-10': '08','KC-767': '08',
                  'ZHISHENGJI': '09',
                  'YUYING': 'Other-YY',
                  'EA-6B': 'Other-TZ','U-125': 'Other-TZ',
                  'T-38': 'Other-JL','MFI-17': 'Other-JL','PC-21': 'Other-JL',
                  'UNKNOWN': 'Other'})

    if not os.path.exists(out_voc_xml):
        os.makedirs(out_voc_xml)
    visual_object_detection_voc(input_voc_xml, out_voc_xml,label_dict)
