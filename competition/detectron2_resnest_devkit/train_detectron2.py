import argparse
import os

from detectron2.utils.logger import setup_logger
from iobjectspy_tools import register_all_pascal_voc, get_classname, get_class_num

setup_logger()

curr_dir = os.path.dirname(os.path.abspath(__file__))

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name,
                         ml_set_tracking_path, experiment_id, ml_experiment_tag):
    cfg = get_cfg()
    cfg.merge_from_file(train_config_path)
    cfg.DATASETS.TRAIN = (register_train_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.MODEL.WEIGHTS = weight_path
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    if max_iter == -1:
        pass
    else:
        cfg.SOLVER.MAX_ITER = max_iter
    num_class = get_class_num(train_data_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_class  # get classes from sda
    cfg.OUTPUT_DIR = out_dir

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    try:
        import mlflow as ml
        # 设置mlflow
        ml.set_tracking_uri(ml_set_tracking_path)
        # 通过设置不同的实验id来管理实验,建议这一层级为项目名称，比如：iobjectspy_faster_rcnn_dota
        ml.set_experiment(experiment_id)
        # 通过设置
        ml.set_tag('experiment_id', ml_experiment_tag)
        ml.log_param('lr', cfg.SOLVER.BASE_LR)
        ml.log_param('max_iter', cfg.SOLVER.MAX_ITER)
        ml.log_param('epoch', cfg.SOLVER.IMS_PER_BATCH)
    except:
        pass


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
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_resnest_devkit/configs/my_experiment/faster_cascade_rcnn_R_50_FPN_syncbn_range-scale_1x.yaml',
        help="path to config file",
    )

    parser.add_argument(
        "--weight_path",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_resnest_devkit/data/model/model/resnetst/resnest50_detectron-255b5649.pth',
        help="path to pre training model ",
    )
    parser.add_argument(
        "--out_dir",
        default='/home/data/hou/workspaces/my_knowledge_base/competition/detectron2_resnest_devkit/out/model50',
        help="path to output directory",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100000,
        help="max iter",
    )

    parser.add_argument(
        "--ml_set_tracking_path",
        default="file:///home/data/windowdata/mlruns",
        help="set tracking path",
    )

    parser.add_argument(
        "--experiment_id",
        default="detectron2_dota",
        help="experiment",
    )

    parser.add_argument(
        "--ml_experiment_tag",
        default="dota_splite800_2020_07_01",
        help="experiment tag",
    )

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    train_data_path = args.train_data_path
    train_config_path = args.train_config_path
    weight_path = args.weight_path
    max_iter = args.max_iter
    out_dir = args.out_dir
    ml_set_tracking_path = args.ml_set_tracking_path
    experiment_id = args.experiment_id
    ml_experiment_tag = args.ml_experiment_tag

    data_path_name = train_data_path.split("/")[-1]
    class_names = get_classname(train_data_path)
    register_all_pascal_voc(train_data_path=train_data_path, class_names=class_names)
    register_train_name = data_path_name + '_trainval'

    train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name,
                         ml_set_tracking_path, experiment_id, ml_experiment_tag)
