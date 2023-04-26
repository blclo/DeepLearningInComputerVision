# dataloader extracting the data from the file and splitting it in validation, test and train
import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image, ExifTags
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import matplotlib as plt
import seaborn as sns; sns.set()

class TacoDataset(Dataset):
    # Returns a compatible Torch Dataset object customized for the WasteProducts dataset
    def __init__(self, dataset_path, split_type, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.split_type = split_type
        if self.split_type == 'train':
            anns_file_path = dataset_path + '/' + 'annotations_0_train.json'
        elif self.split_type == 'val':
            anns_file_path = dataset_path + '/' + 'annotations_0_val.json'
        elif self.split_type == 'test':
            anns_file_path = dataset_path + '/' + 'annotations_0_test.json'
        else:
            raise ValueError('split_type must be one of train, val or test')

        # Read the annotations file
        with open(anns_file_path, 'r') as f:
            print('Reading annotations file: ' + anns_file_path)
            annotations = json.load(f)
            self.categories = annotations['categories']
            self.images = annotations['images']
            self.annotations = annotations['annotations']

        '''
        category_names = []
        supercategory_names = []
        supercategory_id_to_name = {}
        images_id_to_category_id = {}
        nr_supercategories = 0

        # Create a dictionary that maps supercategory names to IDs and store all supercategory names
        for category in self.categories:
            category_names.append(category['name'])
            supercategory_name = category['supercategory']
            if supercategory_name not in supercategory_names:
                supercategory_names.append(supercategory_name)
                supercategory_id_to_name[supercategory_name] = nr_supercategories
                nr_supercategories += 1

        # Redefined classification of supercategories
        """ supercategory_id_to_name =
        {'Aluminium foil': 0, 'Battery': 1, 'Blister pack': 2, 'Bottle': 3, 'Bottle cap': 4, 
        'Broken glass': 5, 'Can': 6, 'Carton': 7, 'Cup': 8, 'Food waste': 9, 'Glass jar': 10, 
        'Lid': 11, 'Other plastic': 12, 'Paper': 13, 'Paper bag': 14, 'Plastic bag & wrapper': 15,
        'Plastic container': 16, 'Plastic glooves': 17, 'Plastic utensils': 18, 'Pop tab': 19,
        'Rope & strings': 20, 'Scrap metal': 21, 'Shoe': 22, 'Squeezable tube': 23, 'Straw': 24,
        'Styrofoam piece': 25, 'Unlabeled litter': 26, 'Cigarette': 27}
        """

        ids = []
        for i in range(len(self.annotations)):
            image_id = self.annotations[i]['image_id']
            category_id_img = self.annotations[i]['category_id']
            if image_id not in ids:
                ids.append(image_id)
                images_id_to_category_id[image_id] = category_id_img
        '''
        # category_id -> supercategory
        self.category_id_to_supercategory = {}
        for category in self.categories:
            category_id = category['id']
            supercategory_name = category['supercategory']
            self.category_id_to_supercategory[category_id] = supercategory_name
        '''
        images_id_to_supercategory = {}    
        for i in range(len(self.annotations)):
            image_id = self.annotations[i]['image_id']
            cat_id = images_id_to_category_id[image_id]
            super_cat = category_id_to_supercategory[cat_id]
            images_id_to_supercategory[image_id] = super_cat

            # this is nice but it's wrong because each image has different supercategories in the annotations, and images_id_to_category_id just has the first one
            
        # ex: {0: 'Bottle', 1: 'Carton', 2: 'Carton'...}
        '''

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return the image and the supercategory annotation for the given index
        # Load image and annotations
        image_id = self.images[idx]["id"]
        image_path = self.dataset_path + '/' + self.images[idx]['file_name']

        image = Image.open(image_path)
        # .convert('RGB') Maybe i need it later - for now it works without it
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        if image._getexif():
            exif = dict(image._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180,expand=True)
                if exif[orientation] == 6:
                    image = image.rotate(270,expand=True)
                if exif[orientation] == 8:
                    image = image.rotate(90,expand=True)
        #image = np.asarray(I)

        # Apply the transformation pipeline if it exists
        if self.transform:
            image = self.transform(image)

        annotations = self.get_annotations(image_id)

        # Convert annotations to tensors
        num_objs = len(annotations)
        boxes = []
        areas = []
        iscrowdlist = []
        segmentations = []
        super_cats_in_img = []
        for i in range(num_objs):
            boxes.append(annotations[i][0])
            segmentations = annotations[i][1][0]
            area = annotations[i][2]
            areas.append(area)
            iscrowd = annotations[i][3]
            iscrowdlist.append(iscrowd)
            super_cat = annotations[i][4]
            super_cats_in_img.append(super_cat)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowdlist, dtype=torch.int64)
        segmentations = torch.as_tensor(segmentations, dtype=torch.float32)

        # Create target dictionary
        target = {}
        target['supercats'] = super_cats_in_img
        target['boxes'] = boxes
        target['segmentations'] = segmentations
        target['area'] = areas
        target['iscrowd'] = iscrowd
        target['image_id'] = torch.tensor([image_id])
        target['image_path'] = image_path

        return image, target


    def get_annotations(self, image_id):
        """Returns all annotations (bounding boxes and labels) for a given image_id."""
        image_annotations = []
        for ann in self.annotations:
            if ann['image_id'] == image_id:
                cat_id = ann['category_id']
                super_cat = self.category_id_to_supercategory[cat_id]
                # TODO: convert the cat_id to super category thanks to the code above and append!
                bbox = ann['bbox']
                segmentation = ann['segmentation']
                area = ann['area']
                iscrowd = ann['iscrowd']
                image_annotations.append((bbox, segmentation, area, iscrowd, super_cat))
        return image_annotations

# -------------------------------------- Data loader --------------------------------------

if __name__ == "__main__":
    def collate_fn(batch):
        # Convert the batch of images and annotations to tensors
        images = []
        annotations = []
        for img, annotation in batch:
            images.append(img)
            annotations.append(annotation)
        images = torch.stack(images, dim=0)
        return images, annotations

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Instantiate the dataset and dataloader
    taco_dataset_train = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'train', transform=transform)
    taco_dataloader_train = DataLoader(taco_dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)

    taco_dataset_val = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'val', transform=transform)
    taco_dataloader_val = DataLoader(taco_dataset_val, batch_size=32, shuffle=True, collate_fn=collate_fn)

    taco_dataset_test = TacoDataset(dataset_path = r'/work3/s212725/WasteProject/data', split_type = 'test', transform=transform)
    taco_dataloader_test = DataLoader(taco_dataset_test, batch_size=32, shuffle=True, collate_fn=collate_fn)
