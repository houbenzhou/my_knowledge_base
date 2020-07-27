## ubuntu desktop


docker build   -f  Dockerfile -t ubuntu_desktop_hou:20.04  .

docker build   -f  Dockerfile_cuda -t ubuntu_desktop_hou_1804_cuda102_cudnn7:v2_2020_06_03  .
## 启动命令

docker run --gpus all -d --net=bridge --restart always --name  ubuntu_desktop_hou -d -p 10001:10001 -p 10002:80 -p 10003:5900 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=hou -v /home:/home -v /dev/shm:/dev/shm ubuntu_desktop_hou_1804_cuda102_cudnn7:v2_2020_06_03

参数说明： 
* run : 创建一个新容器并运行一个命令
* --gpus: 以gpu的形式启动
* -d : 后台启动
* --restart ： 容器退出时候重新启动的策略
* --name : 为容器制定名称
* -p : 将容器端口发布到主机
* --env or -e :设置环境变量
* -v ：主机目录：容器目录

## 需要安装的包
1、java
2、conda
3、pycharm
4、ssr
#安装字体
apt install ibus-clutter
apt install ibus-libpinyin









