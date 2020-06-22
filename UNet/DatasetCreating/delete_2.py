import os

dir_name = 'UnetDataset_v1/train/'
normal_files = os.listdir(dir_name)

for normal_file in normal_files:
    if normal_file.endswith('.jpg'):
        splitted_name = normal_file.split('-')
        if splitted_name[len(splitted_name) - 1] == '2.jpg':
            os.remove(dir_name + normal_file)
            os.remove(dir_name + normal_file[:-4] + '-colormask.png')
