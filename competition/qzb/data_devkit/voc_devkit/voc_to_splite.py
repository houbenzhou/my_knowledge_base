import codecs

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from resources_ml.competition.data_devkit.voc_devkit.create_sda_ImageSets_from_xml_images import _save_sda_file, \
    _save_index_file
from multiprocessing import Pool
import cv2
import glob
import os.path as osp
import os
import xml.etree.ElementTree as ET

class DOTAImageSplitTool(object):
    def __init__(self,
                 in_root,
                 out_root,
                 tile_overlap,
                 tile_shape,
                 num_process=8,
                 ):
        self.in_images_dir = osp.join(in_root, 'Images/')
        self.in_labels_dir = osp.join(in_root, 'Annotations/')
        self.out_images_dir = osp.join(out_root, 'Images/')
        self.out_labels_dir = osp.join(out_root, 'Annotations/')
        assert isinstance(tile_shape, tuple), f'argument "tile_shape" must be tuple but got {type(tile_shape)} instead!'
        assert isinstance(tile_overlap,
                          tuple), f'argument "tile_overlap" must be tuple but got {type(tile_overlap)} instead!'
        self.tile_overlap = tile_overlap
        self.tile_shape = tile_shape
        images = glob.glob(self.in_images_dir + '*.jpg')
        labels = glob.glob(self.in_labels_dir + '*.xml')
        image_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], images)]
        label_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], labels)]
        assert set(image_ids) == set(label_ids)
        self.image_ids = image_ids
        if not osp.isdir(out_root):
            os.mkdir(out_root)
        if not osp.isdir(self.out_images_dir):
            os.mkdir(self.out_images_dir)
        if not osp.isdir(self.out_labels_dir):
            os.mkdir(self.out_labels_dir)
        self.num_process = num_process

    def _parse_annotation_single(self, image_id):
        label_dir = osp.join(self.in_labels_dir, image_id + '.xml')
        tree = ET.parse(label_dir)
        # header = s[:2]
        objects = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text

            bbox = obj.find("bndbox")
            bbox_temp = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox=[bbox_temp[0],bbox_temp[1],bbox_temp[2],bbox_temp[1],bbox_temp[2],bbox_temp[3],bbox_temp[0],bbox_temp[3]]
            center = sum(bbox[0::2]) / 4.0, sum(bbox[1::2]) / 4.0
            objects.append({'bbox': bbox,
                            'label': cls,
                            'difficulty': 0,
                            'center': center})
        return  objects

    def _split_single(self, image_id):
        objs = self._parse_annotation_single(image_id)
        image_dir = osp.join(self.in_images_dir, image_id + '.jpg')
        img = cv2.imread(image_dir)
        h, w, _ = img.shape
        w_ovr, h_ovr = self.tile_overlap
        w_s, h_s = self.tile_shape
        for h_off in range(0, max(1, h - h_ovr), h_s - h_ovr):
            if h_off > 0:
                h_off = min(h - h_s, h_off)  # h_off + hs <= h if h_off > 0
            for w_off in range(0, max(1, w - w_ovr), w_s - w_ovr):
                if w_off > 0:
                    w_off = min(w - w_s, w_off)  # w_off + ws <= w if w_off > 0
                objs_tile = []
                for obj in objs:
                    if w_off <= obj['center'][0] <= w_off + w_s -1:
                        if h_off <= obj['center'][1] <= h_off + h_s-1:
                            obj_temp=[]
                            # obj_temp.append(str(obj))
                            # center = sum(obj['bbox'][0::2]) / 4.0, sum(obj['bbox'][1::2]) / 4.0
                            bbox=[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], obj['bbox'][4], obj['bbox'][5],obj['bbox'][6], obj['bbox'][7]]
                            obj_temp.append({'bbox': bbox,
                                     'label': obj['label'],
                                     'difficulty': 0})
                            if obj_temp[0]['bbox'][5]>h_off + h_s:
                                obj_temp[0]['bbox'][5]=h_off + h_s
                            if obj_temp[0]['bbox'][1]<h_off :
                                obj_temp[0]['bbox'][1]=h_off
                            if obj_temp[0]['bbox'][2]>w_off + w_s:
                                obj_temp[0]['bbox'][2]=w_off + w_s
                            if obj_temp[0]['bbox'][0]<w_off :
                                obj_temp[0]['bbox'][0]=w_off
                            # if obj_temp['bbox'][5]>h_off + h_s:
                            #     obj_temp['bbox'][5]=h_off + h_s
                            # if obj_temp['bbox'][1]<h_off :
                            #     obj_temp['bbox'][1]=h_off
                            # if obj_temp['bbox'][2]>w_off + w_s:
                            #     obj_temp['bbox'][2]=w_off + w_s
                            # if obj_temp['bbox'][0]<w_off :
                            #     obj_temp['bbox'][0]=w_off
                            objs_tile.append(obj_temp[0])
                if len(objs_tile) > 0:
                    img_tile = img[h_off:h_off + h_s, w_off:w_off + w_s, :]
                    save_image_dir = osp.join(self.out_images_dir, f'{image_id}_{w_off}_{h_off}.jpg')
                    save_label_dir = osp.join(self.out_labels_dir, f'{image_id}_{w_off}_{h_off}.xml')
                    cv2.imwrite(save_image_dir, img_tile)

                    # for obj in objs_tile:
                    #     px, py = obj["bbox"][0::2], obj["bbox"][1::2]
                    #     px = map(lambda x: str(x - w_off), px)
                    #     py = map(lambda x: str(x - h_off), py)
                    #     bbox_tile = sum([*zip(px, py)], ())
                    with codecs.open(save_label_dir, "w", "utf-8") as xml:
                        xml.write('<annotation>\n')
                        xml.write('\t<folder>' + 'VOC' + '</folder>\n')
                        xml.write('\t<filename>' + f'{image_id}_{w_off}_{h_off}.jpg' + '</filename>\n')
                        xml.write('\t<size>\n')
                        xml.write('\t\t<width>' + str(self.tile_shape[0]) + '</width>\n')
                        xml.write('\t\t<height>' + str(self.tile_shape[0]) + '</height>\n')
                        xml.write('\t\t<depth>' + str(3) + '</depth>\n')
                        xml.write('\t</size>\n')
                        xml.write('\t\t<segmented>0</segmented>\n')
                        for obj in objs_tile:
                                xml.write('\t<object>\n')
                                xml.write('\t\t<name>' + obj["label"] + '</name>\n')
                                xml.write('\t\t<pose>Unspecified</pose>\n')
                                xml.write('\t\t<truncated>0</truncated>\n')
                                xml.write('\t\t<difficult>0</difficult>\n')
                                xml.write('\t\t<bndbox>\n')
                                xml.write('\t\t\t<xmin>' + str(int(obj["bbox"][0]-w_off)) + '</xmin>\n')
                                xml.write('\t\t\t<ymin>' + str(int(obj["bbox"][1]-h_off)) + '</ymin>\n')
                                xml.write('\t\t\t<xmax>' + str(int(obj["bbox"][2]-w_off)) + '</xmax>\n')
                                xml.write('\t\t\t<ymax>' + str(int(obj["bbox"][5]-h_off)) + '</ymax>\n')
                                xml.write('\t\t</bndbox>\n')
                                xml.write('\t</object>\n')

                        xml.write('</annotation>')

    def split(self):
        with Pool(self.num_process) as p:
            p.map(self._split_single, self.image_ids)


if __name__ == '__main__':
    input_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\tzb2_airplane_voc_modify_v2_20210806\voc'
    output_splite_data=r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\tzb2_airplane_voc_modify_v2_20210806\voc_600'
    tile_shape=(600, 600)
    tile_overlap=(300, 300)
    trainsplit = DOTAImageSplitTool(input_data,
                                    output_splite_data,
                                    tile_overlap=tile_overlap,
                                    tile_shape=tile_shape)
    trainsplit.split()
    source_images = os.path.join(output_splite_data,'Images')
    source_label = os.path.join(output_splite_data,'Annotations')
    out_main_path = os.path.join(output_splite_data, "ImageSets", "Main")
    sda_file=os.path.join(output_splite_data,os.path.basename(output_splite_data)+'.sda')
    _save_sda_file(source_images,source_label,sda_file,tile_shape=tile_shape)
    _save_index_file(out_main_path, source_images)

    # trainsplit = DOTAImageSplitTool(r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\test\voc',
    #                                 r'E:\workspaces\iobjectspy_master\resources_ml\out\tainzhibei2\test\voc_splite',
    #                                 tile_overlap=(400, 400),
    #                                 tile_shape=(800, 800))
    # trainsplit.split()











