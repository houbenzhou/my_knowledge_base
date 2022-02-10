import os


def meger_file_txt(input_file_path1, input_file_path2,output_file_path):
    file_names = os.listdir(input_file_path1)
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    for file_name in file_names:
        input_list1=[]
        input_list2 = []
        output_list = []
        input_file1=os.path.join(input_file_path1,file_name)
        input_file2 = os.path.join(input_file_path2, file_name)
        outfile = os.path.join(output_file_path, file_name)
        with open(input_file1,  "r", encoding="utf-8", errors="ignore") as files:
            try:
                while True:
                    mystr = files.readline()
                    if not mystr:
                        # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                        break
                    input_list1.append(mystr)
            except IOError:
                print(IOError)
        with open(input_file2,  "r", encoding="utf-8", errors="ignore") as files:
            try:
                while True:
                    mystr = files.readline()
                    if not mystr:
                        # 读到数据最后跳出，结束循环。数据的最后也就是读不到数据了，mystr为空的时候
                        break
                    input_list2.append(mystr)
            except IOError:
                print(IOError)

        with open(outfile,  "w", encoding="utf-8", errors="ignore") as files:
            for i in input_list1:
                files.write(i)
            for j in input_list2:
                files.write(j)
        print('')
        pass
            # file_list1 = input_file1.readlines()
            # file_list2 = input_file2.readlines()




if __name__ == '__main__':
    input_file_path1=r'E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer_result'
    input_file_path2=r'E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer_result'
    output_file_path=r'E:\workspaces\iobjectspy_master\resources_ml\competition\out\airport_test_infer_merge'
    meger_file_txt(input_file_path1,input_file_path2,output_file_path)


