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

import xlrd
import xlwt
from xlutils.copy import copy


def build_xls(out_xls):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('test_auto')
    ws.write(1, 0, '水体连通域个数')
    ws.write(2, 0, '水体图片数量')
    ws.write(3, 0, '道路连通域个数')
    ws.write(4, 0, '道路图片数量')

    wb.save(out_xls)


def write_xls_water(num, count, out_xls):
    for str_ind, str_i in enumerate(num):
        rb = xlrd.open_workbook(out_xls)
        wb = copy(rb)
        sheet = wb.get_sheet(0)
        sheet.write(1, str_ind + 1, int(str_i))
        sheet.write(2, str_ind + 1, int(count[str_ind]))

        os.remove(out_xls)
        wb.save(out_xls)
