echo "This container's port is 1080 8080 !"
docker rm -f v2ray_server
docker run --restart always --privileged -d   -p 9851:9851 --name v2ray_server  -v /etc/v2ray/config.json:/etc/v2ray/config.json houbenzhou/v2ray
