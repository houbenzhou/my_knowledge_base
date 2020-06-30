1.在终端激活mlflow安装的conda环境，mlflow ui，根据提示浏览器进入 localhost:5000，查看记录结果

2.其记录的结果会在所运行python文件同目录下的mlruns存储，如果网页端没有看到记录结果，多半为延迟，需要手动操作

3.假设mlruns的存储位置为： ‘/home/lyw/mlruns’，执行操作 mlflow ui --backend-store-uri="file:///home/data/windowdata/mlruns"；目录不同替换存储位置即可，其他不变，刷新网页看到结果

4.其他网页端的操作，可查看“mlflow简单试用”