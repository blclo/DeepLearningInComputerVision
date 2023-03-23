#!/usr/bin/python

import os
import numpy as np
import glob
from tqdm.notebook import tqdm
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from src.data.dataloader import *
from src.data.dataloader import HotDog_NotHotdog
from src.models.model import Network     

if __name__ == '__main__':
    batch_size = 64
    # -----------------  Load dataset  -----------------
    trainset = HotDog_NotHotdog(train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = HotDog_NotHotdog(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # -----------------  Split Dataset  -----------------
    '''
    test_proportion = 0.4
    val_proportion = 0.1

    train_size = np.round((1 - test_proportion) * (1 - val_proportion), 5)
    val_size = np.round((1 - test_proportion) * val_proportion, 5)
    assert np.allclose(train_size + val_size, 1 - test_proportion), "Proportions of the random split does not add up..."

    # The function then randomly divides the input dataset into two non-overlapping subsets, with sizes given by the specified lengths.
    train_dataset, val_dataset, test_dataset = trainset.random_split([train_size, val_size, test_proportion])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, val_size, test_size])
    batch_size = 64
    test_loader = DataLoader(
                    testset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=3,
                ) for testset in [train_dataset, val_dataset, test_dataset]
    '''
    # -----------------  Set up GPU  -----------------
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -----------------  Define model  -----------------
    model = Network()
    model.to(device)

    #Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # TODO: Print model architecture
    print(model)   
    
    # TODO: Notebook analysing the images

    # -----------------  Train model  -----------------
    # TODO: Define train function with basics
    num_epochs = 5

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        #For each epoch
        train_correct = 0
        running_loss_train, running_loss_test = 0.0, 0.0
        test_loss = 0

        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move the data to the device
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            # Training loss
            running_loss_train += loss
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()
        
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()

        # -----------------  Test model  -----------------
        # Compute the test accuracy
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

        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        print(f"EPOCH [{epoch + 1}/{num_epochs}]: ")
        print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
        print(f"Train loss: {running_loss_train / len(train_loader):.3f}")
        print(f"Test loss: {test_loss / len(test_loader):.3f}")

