
import os
import numpy as np
import cv2
from scipy import interpolate
import matplotlib.pyplot as plt

# Specify the directory you want to use
directory = '/media/yanan/One Touch/ggcnn/bosch_obj/mutiple_obj/'

save_path = '/media/yanan/One Touch/ggcnn/bosch_obj/mutiple_obj/'

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if this is a .npy file
    if filename.endswith('.npy'):
        # Full path to the .npy file
        npy_path = os.path.join(directory, filename)

        # Load the .npy file
        array = np.load(npy_path)

        # Create x and y coordinates
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        xx, yy = np.meshgrid(x, y)
        # Get valid (non-zero) and invalid (zero) indices
        valid_idx = np.where(array > 1)
        invalid_idx = np.where(array <= 1)

        # Interpolate only the invalid values
        interp_values = interpolate.griddata(valid_idx, array[valid_idx],
                            invalid_idx, method='nearest')

        # Create a copy of the original image and replace the invalid values
        # with interpolated values
        inpainted = np.copy(array)
        inpainted[invalid_idx] = interp_values
        inpainted = inpainted.astype(np.float32)

        # Display the original and inpainted images side by side
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(array, cmap='gray')
        # ax[0].set_title('Original Image')
        # ax[1].imshow(inpainted, cmap='gray')
        # ax[1].set_title('Inpainted Image')
        # plt.show()


        # Normalize the data to 0 - 255
        # array = ((inpainted - np.min(inpainted)) * (1/(np.max(inpainted) - np.min(inpainted)) * 255)).astype(np.uint8)

        # Full path to the .png file
        png_path = os.path.join(save_path, f'{os.path.splitext(filename)[0]}'+'_depth'+'.tiff')

        # Save the image using OpenCV
        cv2.imwrite(png_path, inpainted)
