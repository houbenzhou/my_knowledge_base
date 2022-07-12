# 现在的类别
# CLASSES = ('Boeing737', 'Boeing747','Boeing777', 'Boeing787', 'C919', 'A220', 'A321', 'A330','A350', 'ARJ21','other-airplane',
# 				'Passenger','Motorboat','Fishing','Tugboat','Engineering','Liquid','Dry','Warship','other-ship',
# 				'Small','Bus','Cargo','Dump','Van','Trailer','Tractor','Excavator','Truck','other-vehicle',
# 				'Basketball','Tennis','Football',
# 				'Baseball','Intersection','Roundabout','Bridge')
# 实际的类别
# ['Bridge', 'Dry Cargo Ship', 'other-vehicle', 'Van', 'Roundabout', 'Boeing787', 'Intersection', 'Small Car', 'Dump Truck', 'Cargo Truck', 'A220', 'Tugboat', 'other-airplane', 'Boeing737', 'A321', 'A330', 'Motorboat', 'Baseball Field', 'Warship', 'Fishing Boat', 'Football Field', 'Basketball Court', 'Boeing777', 'Liquid Cargo Ship', 'Boeing747', 'Bus', 'Truck Tractor', 'other-ship', 'C919', 'Engineering Ship', 'Excavator', 'Passenger Ship', 'A350', 'Trailer', 'Tractor', 'Tennis Court', 'ARJ21']

# ['Liquid', 'Dry', 'A220', 'Dump', 'other-airplane', 'Cargo', 'Excavator', 'Van', 'Intersection', 'Small', 'Baseball', 'Roundabout', 'Engineering', 'Warship', 'A330', 'Football', 'Tennis', 'Passenger', 'Tugboat', 'Fishing', 'Motorboat', 'Boeing737', 'A321', 'Basketball', 'Bridge', 'A350', 'Boeing777', 'Boeing747', 'Boeing787', 'other-ship', 'Bus', 'ARJ21']

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

def gaofen_xml_2xml(input_gaofen_xml, out_gaofen_xml, label_dict,score_thresh):
    input_gaofen_xml_names = os.listdir(input_gaofen_xml)

    for input_gaofen_xml_name in input_gaofen_xml_names:
        # 获取图片路径，用于获取图像大小以及通道数
        input_gaofen_xml_file = os.path.join(input_gaofen_xml, input_gaofen_xml_name)
        tree = ET.parse(input_gaofen_xml_file)
        root = tree.getroot()

        from lxml.etree import Element, SubElement, tostring

        node_root = Element('annotation')

        node_source = SubElement(node_root, 'source')

        node_filename = SubElement(node_source, 'filename')
        for filename in root.iter('filename'):
            node_filename.text = filename.text

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

        for ob in root.iter('object'):
            for name in ob.iter('name'):
                name = name.text
            for probability in ob.iter('probability'):
                probability = probability.text
            points_list=[]
            for points in ob.iter('points'):
                for point in points.iter('point'):
                    point = point.text
                    points_list.append(point)
                for point in points.iter('point'):
                    point = point.text
                    points_list.append(point)
                for point in points.iter('point'):
                    point = point.text
                    points_list.append(point)
                for point in points.iter('point'):
                    point = point.text
                    points_list.append(point)
                for point in points.iter('point'):
                    point = point.text
                    points_list.append(point)
            if float(probability)>=score_thresh:
                node_object = SubElement(node_objects, 'object')

                node_coordinate = SubElement(node_object, 'coordinate')
                node_coordinate.text = "pixel"

                node_type = SubElement(node_object, 'type')
                node_type.text = "rectangle"

                node_description = SubElement(node_object, 'description')
                node_description.text = "None"

                node_possibleresult = SubElement(node_object, 'possibleresult')
                node_name = SubElement(node_possibleresult, 'name')
                node_name.text = label_dict[name]
                node_probability = SubElement(node_possibleresult, 'probability')
                node_probability.text = probability

                node_points = SubElement(node_object, 'points')
                point1 = SubElement(node_points, 'point')
                point1.text = points_list[0]
                point2 = SubElement(node_points, 'point')
                point2.text = points_list[1]
                point3 = SubElement(node_points, 'point')
                point3.text = points_list[2]
                point4 = SubElement(node_points, 'point')
                point4.text = points_list[3]
                point5 = SubElement(node_points, 'point')
                point5.text = points_list[4]

        xml = tostring(node_root, pretty_print=True, encoding='UTF-8')
        output_gaofen_xml_file=os.path.join(out_gaofen_xml, input_gaofen_xml_name)
        with open(output_gaofen_xml_file, 'wb') as f:
            f.write(xml)
def get_parser():
    parser = argparse.ArgumentParser(description="dota test visual")
    parser.add_argument(
        "--input_gaofen_xml",
        default=r"C:\Users\houbenzhou\Desktop\commidt\mergedotav1_dotav15_dong\gaofen4Task1_results_nms_0001\gaofen4Task1_results_nms_0001",
        help="voc image path",
    )
    parser.add_argument(
        "--out_gaofen_xml",
        default=r"C:\Users\houbenzhou\Desktop\commidt\mergedotav1_dotav15_dong\gaofen4Task1_results_nms_0001\gaofen4Task1_results_nms_0001_modify",
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
    input_gaofen_xml = args.input_gaofen_xml
    out_gaofen_xml = args.out_gaofen_xml
    label_dict=({ 'Boeing737': 'Boeing737','Boeing747': 'Boeing747',
                  'Boeing777': 'Boeing777','Boeing787': 'Boeing787',
                  'C919': 'C919','A220': 'A220',
                  'A321': 'A321','A330': 'A330',
                  'A350': 'A350','ARJ21': 'ARJ21',
                  'other-airplane': 'other-airplane','Passenger': 'Passenger Ship',
                  'Motorboat': 'Motorboat','Fishing': 'Fishing Boat',
                  'Tugboat': 'Tugboat','Engineering': 'Engineering Ship',
                  'Liquid': 'Liquid Cargo Ship','Dry': 'Dry Cargo Ship',
                  'Warship': 'Warship','other-ship': 'other-ship',
                  'Small': 'Small Car','Bus': 'Bus',
                  'Cargo': 'Cargo Truck','Dump': 'Dump Truck',
                  'Van': 'Van','Trailer': 'Trailer',
                  'Tractor': 'Tractor','Excavator': 'Excavator',
                  'Truck': 'Truck Tractor','other-vehicle': 'other-vehicle',
                  'Basketball': 'Basketball Court','Tennis': 'Tennis Court',
                  'Football': 'Football Field','Baseball': 'Baseball Field',
                  'Intersection': 'Intersection','Roundabout': 'Roundabout',
                  'Bridge': 'Bridge'
                  })

    if not os.path.exists(out_gaofen_xml):
        os.makedirs(out_gaofen_xml)
    score_thresh=0.001
    gaofen_xml_2xml(input_gaofen_xml, out_gaofen_xml,label_dict,score_thresh)