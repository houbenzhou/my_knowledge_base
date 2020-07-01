# !/usr/bin/env python3
# coding=utf-8
"""
@author: HouBenzhou
@license: 
@contact: houbenzhou@buaa.edu.cn
@software: 
@desc:
"""
import glob
import fitz
import os

#  python -m pip install pymupdf
def pic2pdf(pic_path, pdf_path):

    pic_names = os.listdir(pic_path)
    for pic_name in pic_names:
        doc = fitz.open()
        pdf_name = pic_name.split(".")[0] + '.pdf'
        for img in sorted(glob.glob(os.path.join(pic_path, pic_name))):  # 读取图片，确保按文件名排序
            print(img)
            imgdoc = fitz.open(img)  # 打开图片
            pdfbytes = imgdoc.convertToPDF()  # 使用图片创建单页的 PDF
            imgpdf = fitz.open("pdf", pdfbytes)
            doc.insertPDF(imgpdf)  # 将当前页插入文档
        pdf_file = os.path.join(pdf_path, pdf_name)
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        doc.save(pdf_file)  # 保存pdf文件
        doc.close()


if __name__ == '__main__':
    pic_path = '/home/data/windowdata/temp/gao/1_img'
    pdf_path = '/home/data/windowdata/temp/gao/1_pdf'
    if not os.path.exists(pdf_path):
        os.mkdir(pdf_path)
    pic2pdf(pic_path, pdf_path)
