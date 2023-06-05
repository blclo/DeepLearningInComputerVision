#!/usr/bin/python
# 

import os
from pathlib2 import Path
import torch.nn.functional as F
import torch
import wandb
import pickle
import time
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.data.rp_dataloader import RegionProposalsDatasetTrain, RegionProposalsDataset
from src.data.data_sampler import BalancedSampler
import torchvision.transforms as transforms
from tqdm import trange, tqdm
from src.models.model import get_model
from src.models.utils import set_seed, load_experiment
import json

# Hyperparameters
batch_size = 12

# Set seed
torch.manual_seed(seed=14)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"INFO - using device: {device}")

# ---------------------------- Load Model Checkpoint ---------------------------- #
experiment_checkpoint   = 'ResNet18-lr1e-05.wd0.001.bz32.seed14/best.ckpt'
# Define experiment path
BASE_PATH = Path(r"/work3/s212725/WasteProject")
experiment_path = BASE_PATH / 'models' / experiment_checkpoint

# Load experiment stored from training
model_name, model, criterion = load_experiment(experiment_path, device=device)
model.eval()

#  ---------------  Datasets  ---------------
# define transforms to resize - warping proposals
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

paths_to_probs = {}
path_test = r"/work3/s212725/WasteProject/data/json/new_corrected_test_region_proposals_3.json"
test_dataset = RegionProposalsDataset(path_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------- Testing loop ---------------- #
with torch.no_grad():
    test_loss = 0
    test_correct = 0
    test_total = 0
    for batch in tqdm(test_loader):
        images, labels, paths = batch
        images, labels = images.to(device), labels.to(device)

        # Forward + backward
        outputs = model(images)
        probs = torch.exp(outputs)
        preds = torch.exp(outputs).topk(1)[1]
        
        for path, prob in zip(paths, probs):
            paths_to_probs[path] = prob.tolist()

        # Compute loss
        batch_loss = criterion(outputs, labels)
        test_loss += batch_loss.item()

        # Compute accuracy
        batch_correct = (preds.squeeze() == labels).sum().item()
        test_correct += batch_correct
        test_total += labels.size(0)

    # Compute average loss and accuracy
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_correct / test_total

    print('Test loss: {:.4f}'.format(avg_test_loss))
    print('Test accuracy: {:.2%}'.format(avg_test_acc))
    
# Specify the path for the new JSON file
output_file_path = "/work3/s212725/WasteProject/data/json/paths_to_probs.json"

# Write the dictionary to a JSON file
with open(output_file_path, 'w') as file:
    json.dump(paths_to_probs, file)