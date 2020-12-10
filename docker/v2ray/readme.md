# 云服务器购买地址
https://bandwagonhost.com/aff.php?aff=60098

# v2ray

https://github.com/v2ray/v2ray-core

# v2ray客户端

https://tlanyan.me/v2ray-clients-download/
# ubuntu20.04客户端
https://qv2ray.net/debian/

# v2ray经常断连
sudo apt install ntpdate

ntpdate time.apple.com

# 镜像启动命令

启动服务端：

> docker run --restart always --privileged -d   -p 9851:9851 --name v2ray_server  -v /etc/v2ray/config.json:/etc/v2ray/config.json houbenzhou/v2ray

启动客户端：

> docker run --restart always --privileged -d   -p 1080:1080  -p 8080:8080 --name v2ray_client  -v /etc/v2ray/config.json:/etc/v2ray/config.json houbenzhou/v2ray




