echo "This container's port is 1080 8080 !"
docker rm -f v2ray_client
docker run --restart always --privileged -d   -p 1080:1080  -p 8080:8080 --name v2ray_client  -v /etc/v2ray/config.json:/etc/v2ray/config.json houbenzhou/v2ray
