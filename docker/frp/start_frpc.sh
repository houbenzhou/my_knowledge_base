docker stop frpc
docker rm frpc
docker run --restart=always --network host -d -v /etc/frp/frpc.ini:/etc/frp/frpc.ini --name frpc houbenzhou/frpc