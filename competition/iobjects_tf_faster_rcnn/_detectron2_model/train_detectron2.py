import os

from .iobjectspy_tools import get_class_num
from .utils.logger import setup_logger
# from detectron2.utils.logger import setup_logger
setup_logger()

curr_dir = os.path.dirname(os.path.abspath(__file__))

# from detectron2.engine import DefaultTrainer
# from detectron2.config import get_cfg
from .engine import DefaultTrainer
from .config import get_cfg


def train_iobjectspy_voc(train_data_path, train_config_path, weight_path, max_iter, out_dir, register_train_name,
                         ml_set_tracking_path, experiment_id, ml_experiment_tag):
    cfg = get_cfg()
    cfg.merge_from_file(train_config_path)
    cfg.DATASETS.TRAIN = (register_train_name,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.MODEL.WEIGHTS = weight_path  # initialize from model zoo
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
