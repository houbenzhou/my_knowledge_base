from __future__ import division

from numpy import double
from openpyxl import load_workbook

wb = load_workbook(filename='SAM99.xlsx')

sheet_ranges = wb['Sheet1']
n=1
while n < 1378:
    n=n+1
    sam99=str(sheet_ranges['A'+str(n)].value)
    try:
        coordinate=sam99.split("(")[1].split(")")[0]
    except IndexError as e:
        print()
    coordinate_x= double(coordinate.split(",")[0])
    coordinate_y= double(coordinate.split(",")[1])
    print(coordinate_x)
    print(coordinate_y)

