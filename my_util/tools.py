import os
from  shutil import copyfile

def copy_file_path(source_path, out_path,prefix=None):
    source_filenames=os.listdir(source_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for source_filename in source_filenames:
        if prefix is None:
            source_file_path = os.path.join(source_path, source_filename)
            out_file_path = os.path.join(out_path, source_filename)
            copyfile(source_file_path, out_file_path)
        elif source_filename.endswith(prefix):
            source_file_path = os.path.join(source_path, source_filename)
            out_file_path=os.path.join(out_path, source_filename)
            copyfile(source_file_path, out_file_path)

def delete_file_path(source_path,prefix=None):
    source_filenames=os.listdir(source_path)

    for source_filename in source_filenames:
        if prefix is None:
            source_file_path = os.path.join(source_path, source_filename)
            os.remove(source_file_path)
        elif source_filename.endswith(prefix):
            source_file_path = os.path.join(source_path, source_filename)
            os.remove(source_file_path)

if __name__ == '__main__':
    # source_path=r''
    # out_path = r''
    # copy_file_path(source_path, out_path,'.json')
    source_path=r''
    delete_file_path(source_path,'.json')




