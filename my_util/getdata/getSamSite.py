import csv

import pandas as pd
from numpy import double

csv_reader = csv.reader(open('sam99wgs84_P.csv', encoding='utf-8'))
for row in csv_reader:
    try:
        print(double(row[9]))
    except :
        print()
# csv_readerX = pd.read_csv('sam99wgs84_P.csv',usecols=[9])
# csv_readerY = pd.read_csv('sam99wgs84_P.csv',usecols=[10])
# n = -1
# for row in [0,1,2,1,4,3]:
#         n=n+1
#         samSiteX = double(csv_readerX[n])
#         # samSiteY = double(csv_readerY.pop("Y")[n])
#         print(samSiteX)

