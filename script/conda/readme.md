## conda创建基础环境
> conda create -n py36_torch_cuda10.1 python=3.6 -y

>> conda activate py36_torch_cuda10.1

> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

##　conda克隆基础环境
> conda create -n detectron2-gpu-clone --clone /home/py36_torch_cuda10