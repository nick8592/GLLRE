import os
import math
from PIL import Image


def calc_LRE(image, x_pos, y_pos, n):
    """
    calculate local relative entropy (LRE) of a pixel (x_pos, y_pos) in a nxn neighborhood
    """
    local_relative_entropy = 0
    mean_pixel_value = calc_mean_pixel_value(image, x_pos, y_pos, n)

    for i in range((-(n - 1) / 2), (n - 1) / 2):
        for j in range((-(n - 1) / 2), (n - 1) / 2):
            pixel_value = image.getpixel((x_pos + i, y_pos + j))
            local_relative_entropy += pixel_value * abs(
                math.log(pixel_value / mean_pixel_value)
            )
    return local_relative_entropy


def calc_mean_pixel_value(image, x_pos, y_pos, n):
    """
    calculate mean gray level value of the pixels in the neighborhood
    """
    total_pixel_value = 0
    for i in range((-(n - 1) / 2), (n - 1) / 2):
        for j in range((-(n - 1) / 2), (n - 1) / 2):
            total_pixel_value += image.getpixel((x_pos + i, y_pos + j))
    mean_pixel_value = total_pixel_value / n ^ 2
    return mean_pixel_value


# ===========================================================================
folder_path = "/home/nick/Documents/dataset/COCO/coco2017/test2017"
images_name_list = os.listdir(folder_path)
images_path = [os.path.join(folder_path, name) for name in images_name_list]

# Load the image
image = Image.open(images_path[2])

# Convert the image to grayscale
gray_image = image.convert("L")

image.show()
gray_image.show()

n = 2  # neighborhood
