# 给定一个整形数组nums和一个整数target，返回这两个数字的索引，使它们加起来为target
def twoSum(nums,target):
    nums_index=[(value,index) for index ,value in enumerate(nums)]
    nums_index.sort()
    begin ,end =0,len(nums_index)-1
    while begin < end:
        curr = nums_index[begin][0] +nums_index[end][0]
        if curr == target:
            return [nums_index[begin][1],nums_index[end][1]]
        elif curr < target:
            begin +=1
        else:
            end -=1




if __name__ == '__main__':
    nums=[3, 2, 4]
    target=6
    twoSum(nums,target)














