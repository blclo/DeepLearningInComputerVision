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
from src.data.dataloader import *
from src.data.dataloader import HotDog_NotHotdog
import torchvision.transforms as transforms
from tqdm import trange, tqdm
from src.models.model import get_model
from src.models.utils import set_seed, load_experiment
import json

# Hyperparameters
batch_size = 32

# Set seed
torch.manual_seed(seed=42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"INFO - using device: {device}")

# ---------------------------- Load Model Checkpoint ---------------------------- #
# Define experiment path
experiment_path = '/zhome/7a/d/164695/deep_learning_cv/P1_1_Classification/src/models/hotdog_ResNet18_lr_0.0001_bs_32_epochs_25/best.ckpt'

# Load experiment stored from training
model_name, model, criterion = load_experiment(experiment_path, device=device)
model.eval()

#  ---------------  Datasets  ---------------
testset = HotDog_NotHotdog(train=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
print(len(test_loader))
# ---------------- Testing loop ---------------- #
with torch.no_grad():
    test_correct = 0
    test_total = 0
    for batch in tqdm(test_loader):
        data, target = batch
        data, target = data.to(device), target.to(device)

        # Forward + backward
        outputs = model(data)
        predicted = outputs.argmax(1).cpu()
        print(predicted)
        target = target.cpu()
        print(target)
        test_correct += (target==predicted).sum().item()

    # Compute average loss and accuracy
    avg_test_acc = test_correct / len(testset)
    print('Test accuracy: {:.2%}'.format(avg_test_acc))
    