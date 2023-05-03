# Author: Yanan Liu
# Date: 02/05/2023 15:23
# Location:
# Version:
# Description: Enter a brief description here.

import os
import numpy as np
# Define the parent folder path
parent_folder_path = '/media/yanan/One Touch/detectron2_data_Ma/detectron2_data/crops graph classification/'

task_architecture = [1, 2]
# Iterate over the subfolders in the parent folder
for subfolder_name in os.listdir(parent_folder_path):
    subfolder_path = os.path.join(parent_folder_path, subfolder_name)
    if os.path.isdir(subfolder_path):
        # If the subfolder is a directory, try to open its 'label.txt' file
        label_file_path = os.path.join(subfolder_path, 'label.txt')
        if os.path.isfile(label_file_path):
            # If the 'label.txt' file exists, open it for reading
            with open(label_file_path, 'r+') as label_file:
                gf_lable = []
                # Process the file contents here
                lines = label_file.readlines()
                l = lines[0].split(' ')
                label=[int(i) for i in l]

                for i in range(len(label)):
                    data=np.load(os.path.join(subfolder_path,str(i)+'.npy'),allow_pickle=True)
                    test_label = data.reshape(-1)[0]['label']
                    if label[i] == 1 and test_label == task_architecture[0]:
                        gf_lable.append(1)
                    else:
                        gf_lable.append(0)

                my_string = ' '.join(map(str, gf_lable))
                lines[1] = my_string + '\n'
                label_file.writelines(lines)

        else:
            # If the 'label.txt' file does not exist in the subfolder
            print('Could not find label file in', subfolder_name)


