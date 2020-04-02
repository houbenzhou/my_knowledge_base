import os
import shutil

from PIL import Image, ImageDraw


def visual_dota(input_data, out_path):
    """
    可视化voc数据集
    :param input_data
    :param out_path
    :return:
    """
    img_path = os.path.join(input_data, 'images')
    label_path = os.path.join(input_data, 'labelTxt')
    label_names = os.listdir(label_path)

    for label_name in label_names:
        # 获取图片路径，用于获取图像大小以及通道数
        img_name = label_name.split(".")[0] + '.png'
        img_file = os.path.join(img_path, img_name)
        label_file = os.path.join(label_path, label_name)

        img = Image.open(img_file)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        file = open(label_file, "r", encoding="utf-8", errors="ignore")
        while True:
            mystr = file.readline()  # 表示一次读取一行
            if not mystr:
                # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                break
            label_file_list = mystr.split(' ')
            if len(label_file_list) >= 8:
                x = []
                y = []
                x.append(float(label_file_list[0]))
                x.append(float(label_file_list[2]))
                x.append(float(label_file_list[4]))
                x.append(float(label_file_list[6]))
                y.append(float(label_file_list[1]))
                y.append(float(label_file_list[3]))
                y.append(float(label_file_list[5]))
                y.append(float(label_file_list[7]))
                xmin = min(x)
                ymin = min(y)
                xmax = max(x)
                ymax = max(y)
                categoty = str(label_file_list[8])
                draw.line((xmin, ymin, xmax, ymin), fill=(255, 0, 0), width=4)
                draw.line((xmax, ymin, xmax, ymax), fill=(255, 0, 0), width=4)
                draw.line((xmax, ymax, xmin, ymax), fill=(255, 0, 0), width=4)
                draw.line((xmin, ymax, xmin, ymin), fill=(255, 0, 0), width=4)
                draw.text((xmin, ymin - 44), categoty, fill="#0000ff")
        img.save(os.path.join(out_path, img_name))


if __name__ == '__main__':
    input_data = '/home/data/hou/workspaces/my_knowledge_base/competition/dota/DOTA_Devkit/example'
    out_path = '/home/data/windowdata/temp/visual_dota训练集可视化'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    visual_dota(input_data, out_path)
