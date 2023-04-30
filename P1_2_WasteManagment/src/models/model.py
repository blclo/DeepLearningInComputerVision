from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet50, inception_v3
from torchvision.models import ResNet18_Weights, ResNet50_Weights, Inception_V3_Weights

def get_model(model_name: str, device, lr: Optional[float] = None, weight_decay=0.9, out_dim=28, freeze=False):
    if model_name == '':
        experiment = torch.load((model_name / 'best.ckpt').as_posix()) # train from checkpoint
        model_name = experiment['model']['name']
    else: # FileNotFoundError:
        experiment = None

    if model_name in {'ResNet18', 'ResNet50', 'Inception3'}:
        # Define loss criterion --> NLLLoss used with LogSoftmax for stability reasons
        criterion = nn.NLLLoss()

        if model_name == 'Inception3':
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
        elif model_name == 'ResNet18':
            model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        elif model_name == 'ResNet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        else:
            raise NotImplementedError("Model type not available...")
        
        if freeze == True: # Freeze weights
            for param in model.parameters():
                param.requires_grad = False

        # Define output layers
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_dim),
            nn.LogSoftmax(dim=1)
        ).to(device)

    if experiment != None: # load from checkpoint
        model.load_state_dict(experiment['state_dict'])

    if lr is not None: # For training mode
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer
    else: # For test mode
        return model, criterion, None