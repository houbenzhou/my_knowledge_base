import argparse
import os
import random

from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
from iobjectspy_tools import register_all_pascal_voc, get_classname, get_class_num

setup_logger()

import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def visual_iobjectspy_voc(train_data_path, train_config_path, image_path, register_val_name, model_path, show_images):
    cfg = get_cfg()
    cfg.merge_from_file(train_config_path)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_path  # initialize from model zoo
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    num_class = get_class_num(train_data_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class  # get classes from sda

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.DATASETS.TEST = (register_val_name,)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode

    # dataset_dicts = get_balloon_dicts(os.path.join(train_dir, 'test'))

    pic_names = os.listdir(image_path)
    balloon_metadata = MetadataCatalog.get("iobjectspy_voc")
    for d in random.sample(pic_names, 3):
        im = cv2.imread(os.path.join(image_path, d))
        outputs = predictor(im)
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)

        v = Visualizer(im[:, :, ::-1],
                       balloon_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs['instances'].to("cpu"))
        cv2.imwrite(os.path.join(show_images, d), v.get_image()[:, :, ::-1])


def get_parser():
    parser = argparse.ArgumentParser(description="train detectron2")
    parser.add_argument(
        "--train_data_path",
        default="/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC",
        help="path to train data directory",
    )

    parser.add_argument(
        "--train_config_path",
        default='/home/data/hou/workspaces/detectron2/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml',
        help="path to configs file",
    )

    parser.add_argument(
        "--input_image_path",
        default='/home/data/windowdata/data/dota/dotav1/dotav1/train_val_splite_800/VOC/Images',
        help="path to input image data directory ",
    )
    parser.add_argument(
        "--model_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/temp/model_final.pth',
        help="path to model path directory ",
    )
    parser.add_argument(
        "--out_visual_images",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/visual',
        help="path to out visual images directory ",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    train_data_path = args.train_data_path
    train_config_path = args.train_config_path
    input_image_path = args.input_image_path
    model_path = args.model_path
    out_visual_images = args.out_visual_images

    if not os.path.exists(out_visual_images):
        os.makedirs(out_visual_images)

    data_path_name = train_data_path.split("/")[-1]

    class_names = get_classname(train_data_path)

    register_all_pascal_voc(train_data_path=train_data_path, class_names=class_names,
                            )
    register_val_name = data_path_name + '_test'

    visual_iobjectspy_voc(train_data_path, train_config_path, input_image_path, register_val_name, model_path,
                          out_visual_images)
