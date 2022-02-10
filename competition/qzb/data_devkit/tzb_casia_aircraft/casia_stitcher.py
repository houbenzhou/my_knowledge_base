import os
import sys
from multiprocessing import Pool

import cv2
stitcher = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
def images_stitcher(input_data,imgs_cluster_key,imgs_cluster_value,output_data):

    imgs = []
    for input_data_imgs_name in imgs_cluster_value:
        img = cv2.imread(os.path.join(input_data, input_data_imgs_name))
        if img is None:
            print("can't read image " + input_data_imgs_name)
            sys.exit(-1)
        imgs.append(img)
    try:
        (status, pano) = stitcher.stitch(imgs)

        if status != cv2.STITCHER_OK:
            print("不能拼接图片"+str(imgs_cluster_key))
            sys.exit(-1)
        print("拼接成功："+str(imgs_cluster_key))
        cv2.imwrite(os.path.join(output_data,imgs_cluster_key+'.jpg'),pano)
    except Exception as e:
        print("图片拼接错误："+str(imgs_cluster_key),e)


if __name__ == '__main__':
    input_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\get_01_05_08_form_casia\Images'
    input_data_imgs_names = os.listdir(input_data)
    output_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\CASIA-Aircraft\temp_10'
    if not os.path.exists(output_data):
        os.makedirs(output_data)
    imgs_clusters = {}
    for input_data_imgs_name in input_data_imgs_names:

        # img = cv2.imread(os.path.join(input_data,input_data_imgs_name))
        input_data_imgs_name_list=input_data_imgs_name.split('_')
        key_name=str(input_data_imgs_name_list[0])+'_'+str(input_data_imgs_name_list[1])+'_'+str(input_data_imgs_name_list[2])+'_'+str(input_data_imgs_name_list[3])
        if imgs_clusters.get(key_name)==None:
            imgs_clusters[key_name]=[input_data_imgs_name]
        else:
            imgs_clusters[key_name].append(input_data_imgs_name)
    p = Pool(3)
    for imgs_cluster in imgs_clusters:
        p.apply_async(images_stitcher,args=(input_data,imgs_cluster,imgs_clusters[imgs_cluster],output_data))
        print('Waiting for all resize subprocesses done...')
    p.close()
    p.join()









