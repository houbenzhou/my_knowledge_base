## jupyterlab cuda cudnn

sudo docker build  -f  Dockerfile -t houbenzhou/jupyterlab_cuda_cudnn:20220929  .


## 启动命令

sudo docker stop jupyterlab_cuda_cudnn

sudo docker rm jupyterlab_cuda_cudnn

docker run --gpus all -d --net=bridge --restart always --name  jupyterlab_cuda_cudnn -d -p 10001:8888 -p 10002:8889 -p 10003:8890 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=hou -v /home:/home -v /dev/shm:/dev/shm jupyterlab_cuda_cudnn_hou:20220928

sudo docker run  -itd --net=bridge --restart always --name  jupyterlab_cuda_cudnn -d -p 10001:8888 -p 10002:8889 -p 10003:8890 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=hou -v /home:/home -v /dev/shm:/dev/shm jupyterlab_cuda_cudnn_hou:20220928

sudo docker exec -it cc92 /bin/bash


参数说明： 
* -e LANG=en_US.UTF-8 中文支持
* run : 创建一个新容器并运行一个命令
* --gpus: 以gpu的形式启动,all指的是容器显示所有显卡
* -d : 后台启动
* --restart ： 容器退出时候重新启动的策略
* --name : 为容器制定名称
* -p : 将容器端口发布到主机
* --env or -e :设置环境变量
* -v ：主机目录：容器目录

## jupyter使用方式
基础镜像为conda安装的包

conda中需要有jupyterlab

/root/anaconda3/bin/python  -m pip install jupyterlab

/root/anaconda3/bin/python  -m   jupyterlab --allow_password_change=False  --notebook-dir=/home --no-browser --allow-root --port 8888 --ip=172.17.0.4












