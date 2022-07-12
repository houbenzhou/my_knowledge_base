import time
# 用for循环计算国王的棋盘格子数量

checkerboard_number = 63
start_time = time.time()
for i in range(checkerboard_number):
    if i == 0:
        sum = 1
    else:
        sum = 2 * sum + 1
print('完成，共耗时{}s，sum的数量为{}'.format(
        time.time() - start_time, sum))

def prove(k,result):
    if (k == 1):
        if (pow(2,1)-1)==1:
            wheatNum = 1
            wheatTotalNum = 1
            return True
        else:
            return False
    else:
        proveOfpreviousOne = prove(k-1,result)
        wheatNum *= 2
        wheatTotalNum += wheatNum
        proveOfCurrentOne = False
        if (result.wheatTotalNum == (Math.pow(2, k) - 1)):
            proveOfCurrentOne = True
        if (proveOfPreviousOne & proveOfCurrentOne):
            return True
        else:
            return False


if __name__=='__main__':
    test=prove(100,True)
    print(test)
