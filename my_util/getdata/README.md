# pygetmap

请使用python 3

下载网络地图（路网图和卫星图）。

目前支持的地图有：
- 谷歌 
- 高德 
- 腾讯



## 2017.09.07更新
1.把协程改为多线程实现，增加兼容，减少复杂性，不需要过高的python版本。

2.增加GCJ的纠偏功能

3.增加链接文件的输出，可用于arcgis的地理配准。输出文件可选择原样输出，或gcj02转wgs84，以及wgs84转gcj02。