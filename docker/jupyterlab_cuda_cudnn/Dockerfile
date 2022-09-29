# ==================================================================
# module list
# ------------------------------------------------------------------
# ubuntu           20.04
# cuda             11.6.2
# cudnn            8
# devel
# jupyter          配置文件

FROM  nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
USER root
ENV SOFTWARE=/root/software
RUN mkdir $SOFTWARE
COPY base $SOFTWARE/
COPY jupyter $SOFTWARE/

RUN rm -rf /root/.pip/ && \
    mkdir -p /root/.pip && \
    mkdir -p /root/.jupyter && \
    cp -r $SOFTWARE/pip.conf  /root/.pip/ && \
    cp $SOFTWARE/jupyter_lab_config.py  /root/.jupyter/  &&\
    cp $SOFTWARE/jupyter_server_config.json  /root/.jupyter/

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get clean && apt-get -y update --fix-missing  && apt-get update && apt-get install -y --no-install-recommends \
        gnupg2 curl ca-certificates openjdk-8-jdk  build-essential \
        apt-utils \
        aria2 \
        wget \
        git \
        vim \
        libssl-dev \
        unzip \
        unrar   \
        net-tools &&  \
        rm -rf /var/lib/apt/lists/* && \
        rm /var/lib/apt/lists/* -vf

















