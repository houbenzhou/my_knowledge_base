# 一、qbz_Dockerfile使用方式
### 1.1 构建docker镜像
docker build  -f  qzb_Dockerfile -t ubuntu_desktop/2004_cuda111_cudnn8_pycharm_idesktop:20210928  .

### 1.2 启动docker镜像
docker run --gpus all -e LANG=en_US.UTF-8  -d --net=bridge --restart always --name  qzb_competition -d -p 10101:10001 -p 10102:80 -p 10103:5900 -e RESOLUTION=1920x1080  -e VNC_PASSWORD=qzb -v /home:/home -v /dev/shm:/dev/shm ubuntu_desktop/2004_cuda111_cudnn8_pycharm_idesktop:20210927


# 二、天智杯docker镜像提交

## 镜像中替换conda，tzb2_add_conda_Dockerfile
docker build  -f  tzb2_add_conda_Dockerfile -t houbenzhou/conda:copy_conda_tzb2  .

## 镜像中添加执行脚本，tzb2_final_Dockerfile





