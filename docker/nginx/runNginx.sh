echo "This nginx's port is 10012, 10018 !"
sudo docker rm -f nginx
sudo docker run -d --name nginx -v /home/data1/hou/docker/nginx/config/nginx.conf:/etc/nginx/nginx.conf  -p 10012:10012 -p 10018:10018 nginx
