import argparse
import os

from detectron2.utils.logger import setup_logger
from iobjectspy_tools import register_all_pascal_voc, get_classname, get_class_num

setup_logger()

curr_dir = os.path.dirname(os.path.abspath(__file__))

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name):
    cfg = get_cfg()
    cfg.merge_from_file(train_config_path)
    cfg.DATASETS.TRAIN = (register_train_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = weight_path  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iter  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
    num_class = get_class_num(train_data_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class  # get classes from sda
    cfg.OUTPUT_DIR = out_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


# tree
# def get_parser():
#     parser = argparse.ArgumentParser(description="train detectron2")
#     parser.add_argument(
#         "--train_data_path",
#         default="/home/data/hou/workspaces/iobjectspy2/resources_ml/example/项目/tree/out/voc",
#         help="path to train data directory",
#     )
#
#     parser.add_argument(
#         "--train_config_path",
#         default='/home/data/hou/workspaces/detectron2/configs/my_experiment/cascade_mask_rcnn_R_50_FPN_1x.yaml',
#         help="path to config file",
#     )
#
#     parser.add_argument(
#         "--weight_path",
#         default='/home/data/hou/workspaces/detectron2/data/model/model/ablations_for_deformable_conv_and_cascade_rcnn/cascade_mask_rcnn_R_50_FPN_3x/model_final_480dd8.pkl',
#         help="path to pre training model ",
#     )
#     parser.add_argument(
#         "--out_dir",
#         default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/2020_05_26/tree/model',
#         help="path to output directory",
#     )
#
#     return parser
# dota
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
        help="path to config file",
    )

    parser.add_argument(
        "--weight_path",
        default='/home/data/hou/workspaces/detectron2/data/model/model/ablations_for_deformable_conv_and_cascade_rcnn/cascade_mask_rcnn_R_50_FPN_3x/model_final_480dd8.pkl',
        help="path to pre training model ",
    )
    parser.add_argument(
        "--out_dir",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_devkit/out/2020_05_26/dota/model',
        help="path to output directory",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="max iter",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    train_data_path = args.train_data_path
    train_config_path = args.train_config_path
    weight_path = args.weight_path
    max_iter = args.max_iter
    out_dir = args.out_dir

    data_path_name = train_data_path.split("/")[-1]
    class_names = get_classname(train_data_path)
    register_all_pascal_voc(train_data_path=train_data_path, class_names=class_names)
    register_train_name = data_path_name + '_trainval'

    train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name)
