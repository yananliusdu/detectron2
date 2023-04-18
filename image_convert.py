from PIL import Image
import os

# Set the folder path where the PNG images are stored
png_folder_path = "/media/yanan/One Touch/color/"

# Set the folder path where the converted JPG images will be saved
jpg_folder_path = "/media/yanan/One Touch/color/"

# Loop through all the files in the PNG folder
for filename in os.listdir(png_folder_path):
    if filename.endswith(".png"):
        # Open the PNG image file
        png_image = Image.open(os.path.join(png_folder_path, filename))

        # Convert the PNG image to JPG format
        jpg_image = png_image.convert('RGB')

        # Save the converted JPG image to the JPG folder
        jpg_image.save(os.path.join(jpg_folder_path, filename.replace(".png", ".jpg")))
