import os
import glob

def remove_files(pattern):
    file_list = glob.glob(pattern)
    for file_path in file_list:
        os.remove(file_path)
