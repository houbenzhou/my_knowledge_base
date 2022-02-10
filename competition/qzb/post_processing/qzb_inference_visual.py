import os
import os

import numpy

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
import random
import argparse
import os
import shutil

from PIL import Image, ImageDraw
thr = 0.95


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def visualise_gt(label_path, pic_path, newpic_path):
    color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    # class_list = ['Boeing737','Boeing747','Boeing777', 'Boeing787', 'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other']
    results = GetFileFromThisRootDir(label_path)
    for result in results:
        f = open(result, 'r')
        lines = f.readlines()
        while '\n' in lines:
            lines.remove('\n')

        if len(lines) == 0:  # 如果为空
            # print('文件为空', result)

            # filepath = os.path.join(pic_path, name.split('.')[0] + '.png')
            filepath = os.path.join(pic_path, name.split('txt')[0] + 'tif')

            img = Image.open(filepath)
            img = img.convert('RGB')
            img.save(os.path.join(newpic_path, result.split('\\')[-1].split('txt')[0] + 'png'), im)
            # cv2.imwrite(os.path.join(newpic_path, result.split('\\')[-1].split('txt')[0] + 'png'), im)

        else:
            boxes = []
            for i, line in enumerate(lines):
                # score = float(line.strip().split(' ')[8])
                name = result.split('\\')[-1]
                box = line.strip().split(',')

                boxes.append(box)
            # print(boxes)
                # boxes.append(box)
            # boxes = np.array(boxes, np.float)
            f.close()
            # filepath = os.path.join(pic_path, name.split('.')[0] + '.png')
            filepath = os.path.join(pic_path, name.split('txt')[0] + 'tif')

            # print(filepath)
            img = Image.open(filepath)
            img = img.convert('RGB')
            im = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)
            # print(im)
            for box in boxes:
                # print(box)
                # score = box[-1]
                class_name = box[-1]
                # class_name = str(class_list.index(class_name))

                bbox = list(map(float, box[1:9]))
                bbox = list(map(int, bbox))
                cv2.circle(im, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
                for i in range(3):
                    cv2.line(im, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=(255,255,255),
                             thickness=2)
                cv2.line(im, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=(255,255,255), thickness=2)

                # cv2.putText(im, box[-1], (bbox[6], bbox[7]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0),
                #             1)  # 0.5是字体大小，2是字体的粗细

                # 书写标签
                cv2.rectangle(im, (bbox[0], bbox[1] - 15), (bbox[0] + 20, bbox[1] - 1), (255, 0, 0),
                              thickness=-1)  # thickness表示线的粗细，等于-1表示填充，颜色为(255, 0, 0)
                cv2.putText(im, ' '.join(box[8:]), (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            1)  # 0.5是字体大小，2是字体的粗细
                # if ' '.join(box[8:]) == 'Dry Cargo Ship':
                #     print("05", filepath)
            cv2.imwrite(os.path.join(newpic_path, result.split('\\')[-1].split('.')[0] + '.png'), im)
            # cv2.imwrite(os.path.join(newpic_path, result.split('\\')[-1].split('txt')[0] + 'png'), im)

if __name__ == '__main__':

    # root = r'F:\Code\AerialDetection\zz_data_process\zk_pro\dota_visual'
    pic_path = r'E:\data\0_job_logging\3_competition\7_强智杯\0_qzb_data\worldview05m_test\aircraft_test'  # 样本图片路径
    label_path = r'E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer'  # DOTA标签的所在路径
    newpic_path = r'E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer_visual'  # 可视化保存路径
    if not os.path.isdir(newpic_path):
        os.makedirs(newpic_path)
    visualise_gt(label_path, pic_path, newpic_path)



