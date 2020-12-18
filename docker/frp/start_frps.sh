docker stop frps
docker rm frps
docker run --restart=always --network host -d -v /etc/frp/frps.ini:/etc/frp/frps.ini --name frps houbenzhou/frps