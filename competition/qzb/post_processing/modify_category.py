import os

def modify_category(input_data,output_data):
    """
    修改模型的推理结果

    :param input_data  模型推理结果保存路径
    :param output_data  通过一定的规则修正推理结果的保存路径

    """
    file_names = os.listdir(input_data)
    for file_name in file_names:
        input_file_path = os.path.join(input_data, file_name)
        input_file = open(input_file_path, "r", encoding="utf-8", errors="ignore")
        one_file_all_boxs_list=[]
        try:
            while True:
                mystr = input_file.readline()  # 表示一次读取一行
                if not mystr:
                    # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                    break
                list_bbox = mystr.split(',')
                x = []
                y = []
                x.append(float(list_bbox[1]))
                x.append(float(list_bbox[3]))
                x.append(float(list_bbox[5]))
                x.append(float(list_bbox[7]))
                y.append(float(list_bbox[2]))
                y.append(float(list_bbox[4]))
                y.append(float(list_bbox[6]))
                y.append(float(list_bbox[8]))
                xmin = min(x)
                ymin = min(y)
                xmax = max(x)
                ymax = max(y)
                # 规则修改的位置
                category_name = list_bbox[9].strip()
                if ((ymax-ymin)>=10) & (category_name=="00"):
                    category_name = category_name
                elif (ymax-ymin)>0:
                    pass
                outline = str(list_bbox[0]) + ',' + str(list_bbox[1]) + ',' + str(list_bbox[2]) + ',' + str(
                    list_bbox[3]) + ',' + str(
                    list_bbox[4]) + ',' + str(list_bbox[5]) + ',' + str(list_bbox[6]) + ',' + str(
                    list_bbox[7]) + ',' + str(list_bbox[8]) + ',' + str(
                    category_name)


                one_file_all_boxs_list.append(outline)

        except IOError:
            print(IOError)

        input_file.close()
        output_file_path = os.path.join(output_data, file_name)
        out_file = open(output_file_path, "w", encoding="utf-8", errors="ignore")
        for bbox in one_file_all_boxs_list:
            out_file.write(bbox + '\n')



if __name__ == '__main__':
    input_data=r"E:\workspaces\iobjectspy_master\resources_ml\competition\out\input_inference_result"
    output_data=r"E:\workspaces\iobjectspy_master\resources_ml\competition\out\output_inference_result"
    if not os.path.exists(output_data):
        os.mkdir(output_data)

    modify_category(input_data,output_data)

