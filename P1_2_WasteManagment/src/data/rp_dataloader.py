# dataloader extracting the data from the file and splitting it in validation, test and train
import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import matplotlib as plt
import seaborn as sns; sns.set()
from src.data.dataloader import *
import re
import cv2

class RegionProposalsDataset(Dataset):
    # Returns a compatible Torch Dataset object customized for the WasteProducts dataset
    def __init__(self, dataset_path, transform=None):
        # define path for json file with the proposals
        self.dataset_path = dataset_path
        self.transform = transform

        # Read the annotations file
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        self.paths_to_label = {}
        self.ids_to_path = {}
        for i, (path_to_crop, label) in enumerate(data.items()):
            self.paths_to_label[path_to_crop] = label
            self.ids_to_path[i] = path_to_crop

    def __len__(self):
        return len(self.ids_to_path)

    def __getitem__(self, indices):
        batch_images = []
        batch_labels = []
        batch_paths = []

        for index in indices:
            path = self.ids_to_path[index]

            label = self.paths_to_label[path]
            # open the image
            image = Image.open(path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            batch_images.append(image)
            batch_labels.append(label)
            batch_paths.append(path)

        return batch_images, batch_labels, batch_paths
