#!/usr/bin/python
# 

import os
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import time
from tqdm import trange

from src.models.model import Network

# check GPU or CPU
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# applicable transforms to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # convert image to a PyTorch Tensor
    transforms.RandomHorizontalFlip(p=1),
    transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)
])

# dataset
batch_size = 64
trainset_transformed = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader_t = DataLoader(trainset_transformed, batch_size=batch_size, shuffle=True, num_workers=1)
testset_transformed = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader_t = DataLoader(testset_transformed, batch_size=batch_size, shuffle=False, num_workers=1)

# model definition
model = Network()
model.to(device)

# initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# get the first minibatch
data = next(iter(train_loader_t))[0].cuda()
# try running the model on a minibatch
print('Shape of the output from the convolutional part', model.convolutional(data).shape)
model(data); #if this runs the model dimensions fit

num_epochs = 5

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    running_loss_train, running_loss_test = 0.0, 0.0
    test_loss = 0
    for minibatch_no, (data, target) in tqdm(enumerate(train_loader_t), total=len(train_loader_t)):
        data, target = data.to(device), target.to(device)
        #Zero the gradients computed for each weight
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(data)
        #Compute the loss
        loss = F.nll_loss(torch.log(output), target)
        # training loss
        running_loss_train += loss
        #Backward pass through the network
        loss.backward()
        #Update the weights
        optimizer.step()
        
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target==predicted).sum().cpu().item()

    #Comput the test accuracy
    test_correct = 0


    for data, target in test_loader_t:
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
        
    train_acc = train_correct/len(trainset_transformed)
    test_acc = test_correct/len(testset_transformed)

    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
    print(f"Train loss: {running_loss_train / len(train_loader_t):.3f}")
    print(f"Test loss: {test_loss / len(test_loader_t):.3f}")