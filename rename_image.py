
import os

folder_path = "/home/yanan/detectron2/out/2023_04_17_16_13_47/color" # replace with the path to your folder
file_extension = ".png" # replace with the extension of your image files

counter = 72
for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        new_filename = str(counter) + file_extension
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        counter += 1
