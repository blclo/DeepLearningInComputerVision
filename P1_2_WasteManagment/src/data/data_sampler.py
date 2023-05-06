import random
import numpy as np
from torch.utils.data.sampler import Sampler
import random
import os
from pathlib2 import Path
import torch.nn.functional as F
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.data.rp_dataloader import RegionProposalsDataset
import json
    

class BalancedSampler(Sampler):
    def __init__(self, data, pos_fraction=0.25, batch_size=32):
        self.data = data
        self.pos_fraction = pos_fraction
        self.batch_size = batch_size
        self.positive_samples_idxs = []
        self.negative_samples_idxs = []
        
        # Split samples into positive and negative 
        for i, (path_to_crop, label) in enumerate(data.items()):
            if label == 28:
                self.negative_samples_idxs.append(i)
            else:
                self.positive_samples_idxs.append(i)
                
        # Calculate the number of positive and negative samples in each batch
        self.num_pos_per_batch = int(self.pos_fraction * self.batch_size)
        self.num_neg_per_batch = self.batch_size - self.num_pos_per_batch
        
        # Calculate the number of batches needed to include all samples
        self.num_batches = int(np.ceil(len(self.data) / self.batch_size))
        
        # Create a list of batch indices
        self.batch_indices = []
        for i in range(self.num_batches):
            print(f"Processing batch {i}, out of {self.num_batches}")
            # Randomly sample positive indices
            pos_batch_indices = np.random.choice(self.positive_samples_idxs, size=self.num_pos_per_batch, replace=False)
            # Randomly sample negative indices
            neg_batch_indices = np.random.choice(self.negative_samples_idxs, size=self.num_neg_per_batch, replace=False)
            # Concatenate positive and negative indices to form batch
            batch_indices = np.concatenate([pos_batch_indices, neg_batch_indices])
            # Shuffle the batch indices
            np.random.shuffle(batch_indices)
            # Add the batch indices to the list of batch indices
            self.batch_indices.append(batch_indices)
    
    # provides a way to iterate over indices of dataset elements 
    # returns a list of idxs for iters of each batch
    def __iter__(self):
        # Shuffle the list of batch indices
        np.random.shuffle(self.batch_indices)
        # Yield each batch
        for batch_indices in self.batch_indices:
            yield batch_indices.tolist()
            
    def __len__(self):
        # This method returns the number of samples in the dataset so that the data loader
        # knows how many batches it needs to iterate through.
        return len(self.data)