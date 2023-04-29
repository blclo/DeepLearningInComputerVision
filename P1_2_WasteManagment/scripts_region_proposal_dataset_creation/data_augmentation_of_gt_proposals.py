# dataloader extracting the data from the file and splitting it in validation, test and train
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import re
import cv2


if __name__ == '__main__':
    main_proposals_path = r"C:\Users\carol\deep_learning_in_cv\P1_2_WasteManagment\region_3_proposals.json"
    # load json file in main_proposals_path
    with open(main_proposals_path, 'r') as f:
        data = json.load(f)

    # create a dictionary with the paths to the crops and the labels
    transformed_gt_crops_with_labels = {}
    
    for path_to_crop, label in data.items():
        pattern = r'^/work3/s212725/WasteProject/data/crops/batch_\d+/[0-9]+\d_gt\d+\.jpg$'
        match = re.match(pattern, path_to_crop)
        if match:
            print(f"This path has match as a gt patch: {path_to_crop}")
            image = Image.open(path_to_crop)
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
            ])
            for i in range(10):
                image = transform(image)
                path_transformed_gt = path_to_crop[:-4] + "_augmented_" + str(i) + ".jpg"
                cv2.imwrite(path_transformed_gt, image)
                transformed_gt_crops_with_labels[path_transformed_gt] = label

    json_object = json.dumps(transformed_gt_crops_with_labels, indent=4)
    with open(f"gt_augmented_proposals.json", "w") as outfile:
        outfile.write(json_object)
       
