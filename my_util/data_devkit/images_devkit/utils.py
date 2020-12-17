# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import os
import shutil


def copy_allfiles(src, dest, file_name_path=None):
    """
    copy所有文件到另一个文件夹

    | 如果file_name_path是None，src为文件夹，则将src文件夹拷贝到dest文件夹
    | 如果file_name_path是None，src为文件，则将src文件拷贝到dest文件夹
    | 如果file_name_path不是None，src必须为文件夹，从file_name_path中获取src文件夹中的文件名，并将相应的文件拷贝到dest文件夹

    :param src: 原文件夹
    :type src: str
    :param dest: 目标文件夹
    :type dest: str
    :param file_name_path: 获取文件的file_name
    :type file_name_path: None or str
    """
    if file_name_path is None:
        if os.path.isdir(src):
            src_files = os.listdir(src)
            for file_name in src_files:
                full_file_name = os.path.join(src, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, dest)
        else:
            shutil.copy(src, dest)
    else:
        src_files = os.listdir(file_name_path)
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)
