from src.data.dataloader import *
import torchvision.datasets as datasets
import torch
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import matplotlib.patches as patches
import cv2
import numpy as np
import skimage
import selective_search
import json

def collate_fn(batch):
        # Convert the batch of images and annotations to tensors
        images = []
        annotations = []
        for img, annotation in batch:
            images.append(img)
            annotations.append(annotation)
        images = torch.stack(images, dim=0)
        return images, annotations

def from_tensor_to_tuple(bbox_tensor):
    # convert to tuple
    x_min, y_min, width, height = bbox_tensor.tolist()
    x_max, y_max = x_min + width, y_min + height
    bbox_tuple = (int(x_min), int(y_min), int(x_max), int(y_max))
    return bbox_tuple

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou

def resize_image(image):
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
        ratio = 1
        return image, ratio

    # Resize image with new size and preserve aspect ratio
    resized_image = image.resize(new_size, Image.ANTIALIAS)

    return resized_image, ratio

# define main
if __name__ == '__main__':
    classes = {'Background':28 , 'Aluminium foil': 0, 'Battery': 1, 'Blister pack': 2, 'Bottle': 3, 'Bottle cap': 4, 
        'Broken glass': 5, 'Can': 6, 'Carton': 7, 'Cup': 8, 'Food waste': 9, 'Glass jar': 10, 
        'Lid': 11, 'Other plastic': 12, 'Paper': 13, 'Paper bag': 14, 'Plastic bag & wrapper': 15,
        'Plastic container': 16, 'Plastic glooves': 17, 'Plastic utensils': 18, 'Pop tab': 19,
        'Rope & strings': 20, 'Scrap metal': 21, 'Shoe': 22, 'Squeezable tube': 23, 'Straw': 24,
        'Styrofoam piece': 25, 'Unlabeled litter': 26, 'Cigarette': 27}
        
    # ------------------- Import the dataset -------------------
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Instantiate the dataset and dataloader
    dataset_train = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'train', transform=None)
    taco_dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)

    dataset_test = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'test', transform=transform)
    taco_dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True, collate_fn=collate_fn)

    dataset_val = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'val', transform=transform)
    taco_dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True, collate_fn=collate_fn)

    dataset_path = r'/work3/s212725/WasteProject/data'
    anns_file_path = dataset_path + '/' + 'annotations.json'

    # Read annotations
    with open(anns_file_path, 'r') as f:
        print('Reading annotations file: ' + anns_file_path)
        annotations = json.load(f)
        images = annotations['images']

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    crops_with_labels = {}

    print(f"If the length of the train dataset is {len(dataset_train)} and I am getting 5 proposals per image, my total number of proposals should be {len(dataset_train)*5}")
    for i in range(len(dataset_train)):
        print("Processing Image: " + str(i) + " of " + str(len(dataset_train)) + " images")
        image, anns = dataset_train[i]
        gt_bboxs = anns['boxes'] # Ground Truth boxes in the image
        image_id = anns['image_id']
        super_cats = anns['supercats']
        file_name = images[anns['image_id'].item()]['file_name']
        
        #image_path = dataset_path + '/' + file_name
        resized_image, ratio = resize_image(image)
        image = np.array(resized_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = selective_search.selective_search(image, mode='fast')
        boxes_filter = selective_search.box_filter(boxes, min_size=1, topN=100)
        #print(f"Number of proposals: {len(boxes_filter)}")
        for z in range(len(boxes_filter)):
            assigned_iou = False
            crop = boxes_filter[z]
            # crop image based on crop dimensions
            crop_image = image[crop[1]:crop[3], crop[0]:crop[2]]
            # save cropped image with initial file_name + crop number
            crop_path = f"/work3/s212725/WasteProject/data/crops/{file_name[:-4]}_{z}.jpg"
            if not os.path.exists(os.path.dirname(crop_path)):
                os.makedirs(os.path.dirname(crop_path))
            cv2.imwrite(crop_path, crop_image)
            crops_gt = []
            cats_gt = []

            for j in range(len(anns['boxes'])):
                # compare intersection over union
                gt = from_tensor_to_tuple(anns['boxes'][j])
                gt_resized = tuple(int(ratio * x) for x in gt)
                # crop gt based on gt dimensions
                gt_crop = image[gt_resized[1]:gt_resized[3], gt_resized[0]:gt_resized[2]]
                # save gt crop also in crops
                gt_crop_path_in_crops = f"/work3/s212725/WasteProject/data/crops/{file_name[:-4]}_gt{j}.jpg"
                cv2.imwrite(gt_crop_path_in_crops, gt_crop)

                # save gt crop in a separate gt_crops dir
                # save cropped gt with initial file_name + crop number
                gt_crop_path = f"/work3/s212725/WasteProject/data/gt_crops/{file_name[:-4]}_{j}.jpg"
                if not os.path.exists(os.path.dirname(gt_crop_path)):
                    os.makedirs(os.path.dirname(gt_crop_path))
                cv2.imwrite(gt_crop_path, gt_crop)
                crops_gt.append(gt_resized)
                
                cat_gt = super_cats[j]
                cats_gt.append(cat_gt)
                crops_with_labels[gt_crop_path_in_crops] = classes[cat_gt]

            max_iou = 0
            for j in range(len(crops_gt)):
                iou = calculate_iou(crops_gt[j], crop)
                if iou > max_iou:
                    max_iou = iou
                    index = j
            if max_iou >= 0.5:
                crops_with_labels[crop_path] = classes[cats_gt[j]]
            else:
                crops_with_labels[crop_path] = classes['Background']

        # save checkpoint every 10 iterations
        if i%100 == 0:
            print(f"Processed {i} images, time to save them")
            json_object = json.dumps(crops_with_labels, indent=4)
            # Create a json file 
            with open(f"region_{i}_proposals.json", "w") as outfile:
                outfile.write(json_object)

            if os.path.isfile("region_{i}_proposals.json"):
                print("File created successfully!")
            else:
                print("Error creating file!")
            print(f"Total number of crops: {len(crops_with_labels)}")


    # create the final json file with the crop path and the label
    json_object = json.dumps(crops_with_labels, indent=4)
    
    # Create a json file 
    with open("region_proposals.json", "w") as outfile:
        outfile.write(json_object)

    if os.path.isfile("region_proposals.json"):
        print("File created successfully!")
    else:
        print("Error creating file!")