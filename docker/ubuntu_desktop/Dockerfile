# ==================================================================
# module list
# ------------------------------------------------------------------
# ubuntu          18.04
# idesktop        10i
# iserver         10i

FROM dorowu/ubuntu-desktop-lxde-vnc
USER root
ENV SOFTWARE=/headless/Supermap/

COPY ./software/geoai/ $SOFTWARE/
# ==================================================================
# 支持中文
# ------------------------------------------------------------------
ENV LANG=zh_CN.utf8
RUN export LANG=$LANG
USER root

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

# ==================================================================
# start script
# ------------------------------------------------------------------
EXPOSE 80
WORKDIR /root
ENV HOME=/home/ubuntu \
    SHELL=/bin/bash
HEALTHCHECK --interval=30s --timeout=5s CMD curl --fail http://127.0.0.1:6079/api/health
ENTRYPOINT ["/startup.sh"]
