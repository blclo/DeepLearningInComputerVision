
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

import skimage.io
import json
import cv2
import skimage
import selective_search
import matplotlib.patches as mpatches

dataset_path = r'/work3/s212725/WasteProject/data'
anns_file_path = dataset_path + '/' + 'annotations.json'


def resize_image(image_path):
    with Image.open(image_path) as image:
        # Get original size
        width, height = image.size

        # Determine which dimension to resize based on maximum dimension
        if width > height and width > 480:
            ratio = 480.0 / width
            new_size = (int(width * ratio), int(height * ratio))
        elif height > width and height > 480:
            ratio = 480.0 / height
            new_size = (int(width * ratio), int(height * ratio))
        else:
            return image

        # Resize image with new size and preserve aspect ratio
        resized_image = image.resize(new_size, Image.ANTIALIAS)

        return resized_image

type = 'train'

# Read annotations
dataset_path = r'/work3/s212725/WasteProject/data'
if type == 'train':
    anns_file_path = dataset_path + '/' + 'annotations_0_train.json'
elif type == 'val':
    anns_file_path = dataset_path + '/' + 'annotations_0_val.json'
elif type == 'test':
    anns_file_path = dataset_path + '/' + 'annotations_0_test.json'
else:
    raise ValueError('type must be train, val or test')

# Read annotations
with open(anns_file_path, 'r') as f:
    print('Reading annotations file: ' + anns_file_path)
    annotations = json.load(f)
    images = annotations['images']

for i in range(len(images)):
    file_name = images[i]['file_name']
    image_path = dataset_path + '/' + file_name
    fig, ax = plt.subplots(figsize=(6, 6))
    resized_image = resize_image(image_path)
    
    image = np.array(resized_image)
    # Propose boxes using selective search
    boxes = selective_search.selective_search(image, mode='fast')
    """
    The function returns a list of filtered boxes, where each box is represented 
    as a tuple of four integers (x1, y1, x2, y2), where (x1, y1) are the coordinates 
    of the top-left corner of the box, and (x2, y2) are the coordinates of the bottom-right
    corner of the box.
    """
    boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=1000)
    # Crop the patches corresponding to each box
    for i, box in enumerate(boxes_filter):
        x1, y1, x2, y2 = box
        patch = image[y1:y2, x1:x2]
        # Save patch as a new image
        patch_path = dataset_path + '/' + 'patches/' + file_name.split('.')[0] + '_' + str(i) + '.jpg'

        # GET PATCH AND COMPUTE IOU TO EXTRACT LABEL
        # Save in a dictionary the image name and the corresponding label
        cv2.imwrite(patch_path, patch)

print(f"All images in the {type} dataset have been processed.")