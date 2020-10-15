## ubuntu desktop

docker build  -f  Dockerfile -t ubuntu_desktop_hou_2004_cuda111_cudnn8:20.04-lxqt  .


## 启动命令

docker stop ubuntu_desktop_hou

docker rm ubuntu_desktop_hou

docker run --gpus all -d --net=bridge --restart always --name  ubuntu_desktop_hou -d -p 10001:10001 -p 10002:80 -p 10003:5900 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=hou -v /home:/home -v /dev/shm:/dev/shm ubuntu_desktop_hou_2004_cudabase102:v3_2020_10_15

参数说明： 
* run : 创建一个新容器并运行一个命令
* --gpus: 以gpu的形式启动,all指的是容器显示所有显卡
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










