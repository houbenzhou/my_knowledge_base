import os
import shutil

if __name__ == '__main__':
    # train
    # input_data = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data/train'
    # output_data = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/train'
    # test
    input_data = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data/test'
    output_data = '/home/data/hou/workspaces/my_knowledge_base/competition/gaofen/International/ship_detection_sar/data_v1/test'

    output_data_img = os.path.join(output_data, "images")
    output_data_label = os.path.join(output_data, "label_xml")
    if os.path.exists(output_data_img):
        shutil.rmtree(output_data_img)
    if not os.path.exists(output_data_img):
        os.makedirs(output_data_img)
    if os.path.exists(output_data_label):
        shutil.rmtree(output_data_label)
    if not os.path.exists(output_data_label):
        os.makedirs(output_data_label)
    file_names = os.listdir(input_data)
    for file_name in file_names:
        if file_name.split(".")[-1] == "xml":
            f1 = os.path.join(input_data, file_name)
            f2 = os.path.join(output_data_label, file_name)
        # elif file_name.split(".")[-1] == "tiff":
        else:
            f1 = os.path.join(input_data, file_name)
            f2 = os.path.join(output_data_img, file_name)
        shutil.copy(f1, f2)
