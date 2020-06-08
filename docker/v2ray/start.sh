echo "This container's port is 9851 !"
docker rm -f v2ray9851
docker run --restart always --privileged -d -p 9851:9851   --name v2ray9851  -v /etc/v2ray/config.json:/etc/v2ray/config.json v2fly/v2fly-core