from PIL import Image
import numpy as np
import math

# Open the grayscale image
image = Image.open('/Users/weichenpai/Dataset/coco2017/test2017/000000000106.jpg').convert('L')

# Get the width and height of the image
width, height = image.size

# Define the region size
region_size = 20

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
