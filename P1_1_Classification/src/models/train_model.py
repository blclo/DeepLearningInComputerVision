#!/usr/bin/python

import os
import numpy as np
import glob
from tqdm.notebook import tqdm
from types import SimpleNamespace
import wandb
from tqdm import trange, tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from src.data.dataloader import *
from src.data.dataloader import HotDog_NotHotdog
from src.models.model import get_model

if __name__ == '__main__':
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="deep_learning_in_cv",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "ResNet18",
        "dataset": "hotdog_not_hotdog",
        "epochs": 5,
        },
        name="ResNet18_lr_1e-4_epochs_5",
    )
    # -----------------  Set up GPU  -----------------
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'ResNet18'
    experiment_name = 'resnet18'
    batch_size = 12
    lr=1e-4
    seed = 42

    # Tensorboard writer for logging experiments
    writer = SummaryWriter(f"logs/{experiment_name}")
    # Set seed
    torch.manual_seed(seed)
    
    # -----------------  Load dataset  -----------------
    trainset = HotDog_NotHotdog(train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = HotDog_NotHotdog(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    # -----------------  Define model  -----------------
    model, criterion, optimizer, scheduler = get_model(model_name, lr=lr, device=device)
    model.to(device)
    
    # -----------------  Train model  -----------------
    # TODO: Define train function with basics
    num_epochs = 5

    current_best_loss = torch.inf
    loss_fn = torch.nn.CrossEntropyLoss()

    with trange(num_epochs) as t:
        for epoch in t:
            #For each epoch
            train_correct = 0
            running_loss_train, running_loss_test = 0.0, 0.0
            test_loss = 0

            model.train()
            for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                # Move the data to the device
                data, target = data.to(device), target.to(device)
                # Zero the gradients computed for each weight
                optimizer.zero_grad()
                # Forward pass your image through the network
                output = model(data)
                # Get predictions from log-softmax scores
                preds = torch.exp(output.detach()).topk(1)[1]
                # Compute the loss

                loss = loss_fn(output, target)
                # Training loss
                running_loss_train += loss
                # Backward pass through the network
                loss.backward()
                # Update the weights
                optimizer.step()
            
                # Compute how many were correctly classified
                # Returns the indices of the maximum value of all elements in the input tensor.
                predicted = output.argmax(1)
                train_correct += (target==predicted).sum().cpu().item()

            train_acc = train_correct/len(trainset)
            train_loss = running_loss_train / len(train_loader)
            wandb.log({"Epoch":epoch, "train/acc": train_acc, "train/loss": train_loss})
            print(f"Epoch {epoch} training loss: {train_loss}, accuracy: {train_acc}")
            # -----------------  Test model  -----------------
            # Compute the test accuracy
            model.eval()
            test_correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    output = model(data)
                    # Compute batch loss
                    test_loss += F.nll_loss(output, target).item()
                # Get predicted class
                predicted = output.argmax(1).cpu()
                # Count correct predictions
                target = target.cpu()
                test_correct += (target==predicted).sum().item()

        test_acc = test_correct/len(testset)
        test_loss = test_loss / len(test_loader)
        wandb.log({"Epoch":epoch, "test/acc": test_acc, "test/loss": test_loss})
        print(f"Epoch {epoch} test loss: {test_loss}, accuracy: {test_acc}")
    