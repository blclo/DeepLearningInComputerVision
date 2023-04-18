
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from typing import Tuple
from torchvision.models import resnet18, resnet50, inception_v3
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from typing import Optional

def get_model(model_name: str, lr: float, device):
    if model_name not in ['ResNet18', 'ResNet50', 'Inception3']:
        raise NotImplementedError(f"No such model class exists... {(model_name)}")

    if model_name == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    
    elif model_name == "ResNet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

    else:
        model = inception_v3(pretrained=True)

    # Freeze all layers except the last two blocks
    for name, param in model.named_parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    # Define output layers
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 2),
        nn.LogSoftmax(dim=1)
    ).to(device)
        
    # Define loss criterion + optimizer --> NLLLoss used with LogSoftmax for stability reasons
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
    return model, criterion, optimizer, scheduler