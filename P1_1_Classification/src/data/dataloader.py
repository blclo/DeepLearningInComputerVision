from typing import Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import glob
import PIL.Image as Image


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform=None, size=128, data_path='/dtu/datasets1/02514/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        self.train = train
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        self.convert_tensor = transforms.Compose([transforms.ColorJitter(),
                                      transforms.RandomPerspective(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                     ])
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.convert_tensor(Image.open(image_path))
        
        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 5 images
        if self.train and idx > 0 and idx%5 == 0:
            # Choose another image/label randomly
            mixup_idx = np.random.randint(0, len(self.image_paths)-1)
            mixup_image_path = self.image_paths[mixup_idx]
            mixup_image = self.convert_tensor(Image.open(mixup_image_path))
            c = os.path.split(os.path.split(mixup_image_path)[0])[1]
            mixup_label = self.name_to_label[c]

            # Select a random number from the given beta distribution
            # Mixup the images accordingly
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            X = lam * X + (1 - lam) * mixup_image
            y = lam * y + (1 - lam) * mixup_label
            
        if self.transform:
            X = self.transform(X)
            
        return X, y
    