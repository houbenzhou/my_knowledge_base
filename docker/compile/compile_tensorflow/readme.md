# tensorflow build from source


## Description
此镜像用于编译tensorflow1.15源码
| tensorflow| cuda版本号 | cudnn版本号 | gcc版本号 | bazel版本号 |python版本号 |
| ------:| ------: | :------: |------:| ------: | :------: |
| 1.15 | 10.1 | 7 |7.5 | 0.26 | 3.6 |
| 1.15 | 10.1 | 7 |7.5 | 0.26 | 3.7 |
| 1.15 | 10.1 | 7 |7.5 | 0.26 | 3.8 |
| 1.15 | 10.2 | 7 |7.5 | 0.26 | 3.6 |
| 1.15 | 10.2 | 7 |7.5 | 0.26 | 3.7 |
| 1.15 | 10.2 | 7 |7.5 | 0.26 | 3.8 |
| 1.15 | 11.1 | 8 |7.5 | 0.26 | 3.6 |
| 1.15 | 11.1 | 8 |7.5 | 0.26 | 3.7 |
| 1.15 | 11.1 | 8 |7.5 | 0.26 | 3.8 |


## 打包docker镜像
```
docker build  -f  Dockerfile_cuda101_cudnn7_gcc75_bazel026_py36 -t houbenzhou/tensorflow_compile:cuda101_cudnn7_gcc75_bazel026_py36  .
```
参数说明：
* build ： 通过Dockerfile构建镜像

* --build-arg ： 设置构建镜像时用到的环境变量，上面输入的是代理地址，编译时候需要加上代理才能顺利编译成功。

* --file or -f : Dockerfile的文件名

* --tag  or -t : 镜像名称
## 镜像命名规范

```
houbenzhou/compile_tensorflow:<cuda版本号>_<cudnn版本号>_<gcc版本号>_<bazel版本号>_<python版本号>
```            
例如：
```            
houbenzhou/tensorflow_compile:cuda101_cudnn7_gcc75_bazel026_py36 
``` 

## 启动docker容器
``` 
docker run --gpus all -it -w /tensorflow_src --name from_tensorflow_cuda101_cudnn7_dev -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)"   from_tensorflow_cuda101_cudnn7_dev
``` 

参数说明： 
* run : 创建一个新容器并运行一个命令
* -it : 交互式操作容器
* -w ： 工作空间，启动容器后进入的目录
* --name : 为容器制定名称
* --gpus : 保证容器可以正常调用gpu
* -v ：主机目录：容器目录

## git clone tensorflow 然后切换分支
``` 
https://github.com/tensorflow/tensorflow.git
``` 
``` 
git checkout r1.15
``` 

## 配置构建

``` 
./configure
``` 
## 构建用于创建pip软件包的工具
``` 
bazel build --config=opt --config=v1 --config=cuda //tensorflow/tools/pip_package:build_pip_package
``` 
参数说明： 
* --config=v1 ： 编译v1版本的tensorflow
* --config=cuda : 编译gpu版本的tensorflow

##调整文件在容器外部的所有权
``` 
chown $HOST_PERMS /mnt/tensorflow-version-tags.whl
``` 
## 然后就可以安装使用了















