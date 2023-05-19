
import os

folder_path = r"out/2023_05_15_10_16_09/color" # replace with the path to your folder
file_extension = ".png" # replace with the extension of your image files

counter = 253
for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        new_filename = str(counter) + file_extension
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        counter += 1
