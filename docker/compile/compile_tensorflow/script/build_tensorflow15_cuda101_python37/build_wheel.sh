#! /bin/bash
export http_proxy=http://192.168.13.175:8080 &&\
export https_proxy=http://192.168.13.175:8080 &&\
cp -r /mnt/tensorflowr115.tar.gz /tensorflow_src &&\
cd /tensorflow_src &&\
tar -zxvf tensorflowr115.tar.gz &&\
cd tensorflow &&\
./configure &&\
bazel build --config=opt --config=v1 --config=cuda //tensorflow/tools/pip_package:build_pip_package &&\
chown $HOST_PERMS /mnt/*.whl