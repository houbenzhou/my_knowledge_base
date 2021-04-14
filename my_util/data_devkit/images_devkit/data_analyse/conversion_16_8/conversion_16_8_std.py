import os

from osgeo import gdal
import numpy as np


def Bit8(RawBandData, dMin, dMax, Mean, StdDev, Kn):
    ucMax = Mean + Kn * StdDev
    ucMin = Mean - Kn * StdDev
    k = (dMax - dMin) / (ucMax - ucMin)
    b = (ucMax * dMin - ucMin * dMax) / (ucMax - ucMin)
    if (ucMin <= 0):
        ucMin = 0

    RawBandData = np.select(
        [RawBandData == dMin, RawBandData <= ucMin, RawBandData >= ucMax, k * RawBandData + b < dMin,
         k * RawBandData + b > dMax,
         (k * RawBandData + b > dMin) & (k * RawBandData + b < dMax)],
        [dMin, dMin, dMax, dMin, dMax, k * RawBandData + b], RawBandData)
    return RawBandData


def convert2bit8(InPath, OutFile):
    if not os.path.exists(InPath):
        print('输入路径不存在！')
        return None
    for root, dirs, files in os.walk(InPath):
        for file in files:
            '''筛选tif文件'''
            if not file.split('.')[-1] == 'tiff':
                continue
            '''输入路径的文件名'''
            file_name = os.path.join(root, file)
            '''创建输出路径文件名'''
            # OutFile = os.path.join(root, os.path.splitext(file)[0] + '_Bit8.tif')
            '''文件存在则删除文件重新生成'''
            if os.path.exists(OutFile):
                os.remove(OutFile)
            InTif = gdal.Open(file_name)
            Width = InTif.RasterXSize
            Height = InTif.RasterYSize

            '''跳过8bit'''
            if InTif.ReadAsArray().dtype.name == 'uint8':
                continue

            geoTransform = InTif.GetGeoTransform()
            print('左上角坐标 %f %f' % (geoTransform[0], geoTransform[3]))
            print('像素分辨率 %f %f' % (geoTransform[1], geoTransform[5]))
            '''True Color 1,2,3波段'''
            OutTif = gdal.GetDriverByName('GTiff').Create(OutFile, Width, Height, InTif.RasterCount, gdal.GDT_Byte)
            OutTif.SetProjection(InTif.GetProjection())
            OutTif.SetGeoTransform(geoTransform)

            for i in range(1, int(InTif.RasterCount) + 1):
                RawBand = InTif.GetRasterBand(i)
                RawBandData = RawBand.ReadAsArray()
                Mean = np.mean(RawBandData)
                StdDev = np.std(RawBandData, ddof=1)
                # StdDev = np.std(RawBandData)
                OutBand = Bit8(RawBandData, 0, 255, Mean, StdDev, 2.5)
                OutTif.GetRasterBand(i).WriteArray(OutBand)

            OutTif.FlushCache()
            for i in range(1, int(InTif.RasterCount) + 1):
                OutTif.GetRasterBand(i).ComputeStatistics(False)
            OutTif.BuildOverviews('average', [2, 4, 8, 16, 32])
            del OutTif
    return OutFile


if __name__ == '__main__':
    '''输入路径'''
    # InPath = r"D:\Data\hsd\201911\01\YT"
    curr_dir = os.path.join('E:\\supermap', '2_ai_example', 'demov3(20210310)')

    InPath = os.path.join(curr_dir, 'data', '2_test', '16')
    OutFile = os.path.join(curr_dir, 'data', '2_test','8', 'demo_8.tiff')
    input_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\1.tif'
    out_path = r'E:\supermap\2_ai_example\demov3(20210310)\data\2_test\1_8_std.tif'

    convert2bit8(input_path, out_path)
