# 基本的ubuntu桌面环境

## Description
开放给用户试用的docker镜像,本项目包含了ubuntu、vnc、idesktop、iserver：


## 解压docker镜像

```
docker load -i ubuntu_idesktop_iserver
```


##启动docker镜像(cpu)
```
docker run  -d --net=bridge --name ubuntu_vnc_base_new  -e VNC_PW=hou   -e VNC_RESOLUTION=1920x1080  -p 5901:5901 -p 6901:6901 -p 8888:8888 -v /home:/home   --ipc=host  ubuntu_desktop_hou_cpu:geoai
```
##启动docker镜像(gpu)
```
docker run --gpus all -d --net=bridge --name ubuntu_vnc_base_new  -e VNC_PW=hou  -e VNC_RESOLUTION=1920x1080  -p 5901:5901 -p 6901:6901  -p 8888:8888 -v /home:/home   --ipc=host  ubuntu_desktop_hou_gpu:geoai

```
参数说明： 
* run : 创建一个新容器并运行一个命令
* -d : 后台启动
* --restart ： 容器退出时候重新启动的策略
* --name : 为容器制定名称
* -p : 将容器端口发布到主机
* --env or -e :设置环境变量
* --ipc : 共享内存的方式加速进程间通信，通常用于科学计算
* --runtime=nvidia : gpu的docker需要加上这个参数来启动
* --gpus all 指的是获取所有显卡信息
* -v ：主机目录：容器目录

启动容器镜像时注意：容器内部端口5901是vnc端口，6901是novnc端口，-e VNC_PW用来设置vnc登录密码，-e VNC_RESOLUTION用来设置分辨率

