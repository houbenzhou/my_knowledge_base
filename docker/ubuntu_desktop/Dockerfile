# ==================================================================
# module list
# ------------------------------------------------------------------
# ubuntu           20.04
# java组件         19012-82392
# jdk              8
# lxqt

FROM dorowu/ubuntu-desktop-lxde-vnc:focal-lxqt
USER root
ENV SOFTWARE=/headless/Supermap/

COPY ./software/geoai/ $SOFTWARE/
RUN cp $SOFTWARE/sources.list /etc/apt

# ==================================================================
# software
# ------------------------------------------------------------------

RUN rm -rf /root/.pip/ && \
    mkdir -p /root/.pip && \
    mkdir -p /headless/Supermap/ && \
    mkdir -p /opt/SuperMap/License/&& \
    mkdir -p /opt/SuperMap/iobjects/1010/ && \
    cp $SOFTWARE/pip.conf  /root/.pip/ && \
    cp $SOFTWARE/supermap_any_2020.lic9d  /opt/SuperMap/License/ && \
    cp -r $SOFTWARE/Bin  /opt/SuperMap/iobjects/1010/ && \
    cp $SOFTWARE/pycharm.desktop  /usr/share/applications

