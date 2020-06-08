## ubuntu desktop


docker build   -f  Dockerfile -t ubuntu_desktop_hou:20.04  .

docker build   -f  Dockerfile_cuda -t ubuntu_desktop_hou_1804_cuda102_cudnn7:v2_2020_06_03  .
## 启动命令

docker run --gpus all -d --net=bridge --restart always --name  ubuntu_desktop_hou -d -p 10001:10001 -p 10002:80 -p 10003:5900 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=hou -v /home:/home -v /dev/shm:/dev/shm ubuntu_desktop_hou_1804_cuda102_cudnn7:v2_2020_06_03

## 需要安装的包
1、java
2、conda
3、pycharm
4、ssr









