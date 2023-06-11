import os
import math
import numpy as np
from PIL import Image


def calc_LRE(image, width, height, region_size):
    """
    calculate local relative entropy (LRE) of a pixel (x_pos, y_pos) in a nxn neighborhood
    """
    # Create an empty NumPy array to store the modified pixel values
    modified_array = np.empty((height, width), dtype=np.uint8)

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the region boundaries
            left = max(0, x - region_size // 2)
            upper = max(0, y - region_size // 2)
            right = min(width, x + region_size // 2 + 1)
            lower = min(height, y + region_size // 2 + 1)

            # Extract the region from the image
            region = image.crop((left, upper, right, lower))

            # Calculate the mean of the region
            region_mean = sum(region.getdata()) // (region_size * region_size)

            # Get the pixel value at the current position
            pixel_value = image.getpixel((x, y))

            # Check if the pixel value or region mean is non-positive or zero
            if pixel_value <= 0 or region_mean <= 0:
                modified_array[y, x] = pixel_value
            else:
                # Perform the operations on the pixel value
                result = math.log(abs(pixel_value / region_mean)) * pixel_value

                # Normalize the result to the range of 0 to 255
                normalized_value = max(0, min(255, result))

                # Update the modified array with the normalized value
                modified_array[y, x] = int(normalized_value)

    # Save the modified pixel values as a grayscale image
    modified_image = Image.fromarray(modified_array, mode='L')
    modified_image.save('modified_image.png')

    return modified_array


# ===========================================================================
folder_path = "/Users/weichenpai/Dataset/coco2017/test2017"
images_name_list = os.listdir(folder_path)
images_path = [os.path.join(folder_path, name) for name in images_name_list]

# Load the image
image = Image.open(images_path[5])

# Get the width and height of the image
width, height = image.size

# Print the width and height
print(f"Height: {height}")
print(f"Width: {width}")

# Convert the image to grayscale
gray_image = image.convert("L")
gray_arr = np.asarray(gray_image)

image.show()
gray_image.show()

n = 20  # neighborhood
lre_arr = calc_LRE(gray_image, width, height, n)

print(gray_arr.shape)
print(lre_arr.shape)

# Flatten the images to 1D arrays
I_flat = gray_arr.flatten()
J_flat = lre_arr.flatten()

# Calculate nij, pixel pairs
nij = np.zeros((256, 256), dtype=int)
for i in range(height * width):
    nij[I_flat[i], J_flat[i]] += 1

# Calculate pij, occurence frequency
pij = nij / (height * width)
print(pij.shape)

