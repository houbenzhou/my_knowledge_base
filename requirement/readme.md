# 代理
export http_proxy=http://192.168.13.166:10809
export https_proxy=http://192.168.13.166:10809

export http_proxy=http://192.168.13.175:8080
export https_proxy=http://192.168.13.175:8080

# conda创建基础环境
## conda通过配置文件创建基础环境
> conda env create -f requirements-conda-gpu-pytorch17.yml
## conda通过命令创建基础环境
> conda create -n py36_torch_cuda10.1 python=3.6 -y

> conda activate py36_torch_cuda10.1

> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

> conda install -n pytorch17_gpu_test_cuda101_tensorflow15 /home/data/hou/workspaces/my_knowledge_base/docker/compile_tensorflow/package/conda_package/linux/tensorflow_115_cp36/tensorflow-gpu-1.15.4-py36h39e3cac_1.tar.bz2

