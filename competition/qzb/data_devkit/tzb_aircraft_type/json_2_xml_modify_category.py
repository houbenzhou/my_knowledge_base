import json, codecs, os
from collections import OrderedDict

from tqdm import tqdm
import numpy as np

# 标签路径
labelme_path = r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\train_val_json"  # 原始labelme标注数据路径
saved_path = r"E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\tzb2_airplane_voc_modify_v1_20210804\voc\Annotations"  # 保存路径
os.makedirs(saved_path, exist_ok=True)
label_dict=({ 'A': '02','B': '09','C': '06','D': '00','E': '03','F': '04','G': '03','H': '05','I': 'Other-YY','J': 'Other-KJ','K': 'Other' })

# 获取待处理文件
files = os.listdir(labelme_path)
files = list(filter(lambda x: x.endswith('json'), files))
files = [i.split("/")[-1].split(".json")[0] for i in files]

# 读取标注信息并写入 xml
for json_file_ in tqdm(files):
    json_filename = os.path.join(labelme_path, json_file_ + ".json")
    json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    height, width, channels = json_file['imageHeight'], json_file['imageWidth'], 3
    with codecs.open(os.path.join(saved_path, json_file_ + ".xml"), "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'VOC' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]

            # 通过字典转换类别信息
            label = label_dict[label]

            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
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
