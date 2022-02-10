## 竞赛说明

机器学习资源包包含了SuperMap各产品中机器学习功能所需文件，如示例数据、程序、模型和训练配置文件等，详情请见目录结构。

### 使用方式


### 目录结构

    competition
    ├─data_devkit：对训练数据集进行分析，处理以及可视化;添加一份数据集就新建一个文件夹
    │  │  tzb_aircraft_type：对天智杯飞机的数据进行处理
    │  │  
    │  
    ├─image_maching：地图匹配，以图搜图
    │  
    ├─post_processing：后处理
    ├─rotate_rectangle：旋转矩形
    ├─train_inference_detectron2：基于detectron2的训练以及预测
    │  ├─backbone：预训练权重
    │  ├─config：配置文件用于传参；添加一份数据就添加一个文件夹；配置文件命名算法
    │  ├─tools：训练推理脚本与config保持同名
    │  ├─inference.py：推理入口
    │  ├─train.py：训练入口
    │  ├─inference：推理示例数据
    ├─train_inference_mmdetection：基于mmdetection的训练以及预测
    │  ├─backbone：预训练权重
    │  ├─config：配置文件用于传参
    │  ├─tools：训练推理脚本与config保持同名
    │  ├─inference.py：推理入口
    │  ├─train.py：训练入口
    │  ├─inference：推理示例数据
