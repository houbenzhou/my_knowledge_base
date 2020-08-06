from multiprocessing import Pool
import cv2
import glob
import os.path as osp
import os


class DOTAImageSplitTool(object):
    def __init__(self,
                 in_root,
                 out_root,
                 tile_overlap,
                 tile_shape,
                 num_process=8,
                 ):
        self.in_images_dir = osp.join(in_root, 'images/')
        self.in_labels_dir = osp.join(in_root, 'labelTxt/')
        self.out_images_dir = osp.join(out_root, 'images/')
        self.out_labels_dir = osp.join(out_root, 'labelTxt/')
        assert isinstance(tile_shape, tuple), f'argument "tile_shape" must be tuple but got {type(tile_shape)} instead!'
        assert isinstance(tile_overlap,
                          tuple), f'argument "tile_overlap" must be tuple but got {type(tile_overlap)} instead!'
        self.tile_overlap = tile_overlap
        self.tile_shape = tile_shape
        images = glob.glob(self.in_images_dir + '*.png')
        labels = glob.glob(self.in_labels_dir + '*.txt')
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
        label_dir = osp.join(self.in_labels_dir, image_id + '.txt')
        with open(label_dir, 'r') as f:
            s = f.readlines()
        header = s[:2]
        objects = []
        s = s[2:]
        for si in s:
            bbox_info = si.split()
            assert len(bbox_info) == 10
            bbox = [*map(lambda x: int(x), bbox_info[:8])]
            center = sum(bbox[0::2]) / 4.0, sum(bbox[1::2]) / 4.0
            objects.append({'bbox': bbox,
                            'label': bbox_info[8],
                            'difficulty': int(bbox_info[9]),
                            'center': center})
        return header, objects

    def _split_single(self, image_id):
        hdr, objs = self._parse_annotation_single(image_id)
        image_dir = osp.join(self.in_images_dir, image_id + '.png')
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
                    if w_off <= obj['center'][0] <= w_off + w_s - 1:
                        if h_off <= obj['center'][1] <= h_off + h_s - 1:
                            objs_tile.append(obj)
                if len(objs_tile) > 0:
                    img_tile = img[h_off:h_off + h_s, w_off:w_off + w_s, :]
                    save_image_dir = osp.join(self.out_images_dir, f'{image_id}_{w_off}_{h_off}.png')
                    save_label_dir = osp.join(self.out_labels_dir, f'{image_id}_{w_off}_{h_off}.txt')
                    cv2.imwrite(save_image_dir, img_tile)
                    label_tile = hdr[:]
                    for obj in objs_tile:
                        px, py = obj["bbox"][0::2], obj["bbox"][1::2]
                        px = map(lambda x: str(x - w_off), px)
                        py = map(lambda x: str(x - h_off), py)
                        bbox_tile = sum([*zip(px, py)], ())
                        obj_s = f'{" ".join(bbox_tile)} {obj["label"]} {obj["difficulty"]}\n'
                        label_tile.append(obj_s)
                    with open(save_label_dir, 'w') as f:
                        f.writelines(label_tile)

    def split(self):
        with Pool(self.num_process) as p:
            p.map(self._split_single, self.image_ids)


if __name__ == '__main__':
    trainsplit = DOTAImageSplitTool('/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/train',
                                    '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/split/train',
                                    tile_overlap=(150, 150),
                                    tile_shape=(600, 600))
    trainsplit.split()
    valsplit = DOTAImageSplitTool('/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/val',
                                  '/home/data/windowdata/data/dota/dotav1/dotav1/train_and_val/split/val',
                                  tile_overlap=(150, 150),
                                  tile_shape=(600, 600))
    valsplit.split()
