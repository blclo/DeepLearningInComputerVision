# dataloader extracting the data from the file and splitting it in validation, test and train
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import re
import cv2
import numpy as np
import torchvision.utils as utils

if __name__ == '__main__':
    main_proposals_path = "/work3/s212725/WasteProject/data/json/train_region_proposals.json"
    # load json file in main_proposals_path
    with open(main_proposals_path, 'r') as f:
        data = json.load(f)

    # create a dictionary with the paths to the crops and the labels
    transformed_gt_crops_with_labels = {}
    
    for path_to_crop, label in data.items():
        filename = path_to_crop.split("/")[-2:] # split path by "/", get the last two elements
        filename = "/".join(filename)  # join the last two elements with "/"
        pattern = r'^/work3/s212725/WasteProject/data/crops/batch_\d+/[0-9]+\d_gt\d+\.jpg$'
        match = re.match(pattern, path_to_crop)
        if match:
            print(f"This path has match as a gt patch: {path_to_crop}")
            image = Image.open(path_to_crop)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.5, hue=0.5),
            ])
            for i in range(10):
                image = transform(image)
                path_transformed_gt = r"/work3/s212725/WasteProject/data/augmented_proposals/" + filename[:-4] + "_augmented_" + str(i) + ".jpg"
                if not os.path.exists(os.path.dirname(path_transformed_gt)):
                    os.makedirs(os.path.dirname(path_transformed_gt))
                image.save(path_transformed_gt)
                #print(f"Image nº {i} saved")
                transformed_gt_crops_with_labels[path_transformed_gt] = label
    
    json_object = json.dumps(transformed_gt_crops_with_labels, indent=4)
    with open(f"gt_augmented_proposals.json", "w") as outfile:
        outfile.write(json_object)