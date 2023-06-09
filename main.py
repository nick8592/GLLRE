import os
from PIL import Image

folder_path = "/home/nick/Documents/dataset/COCO/coco2017/test2017"
images_name_list = os.listdir(folder_path)
images_path = [os.path.join(folder_path, name) for name in images_name_list]

# Load the image
image = Image.open(images_path[2])

# Convert the image to grayscale
gray_image = image.convert('L')

image.show()
gray_image.show()
