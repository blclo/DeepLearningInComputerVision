#!/usr/bin/python
# 

import os
from pathlib2 import Path
import torch.nn.functional as F
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.data.rp_dataloader import RegionProposalsDataset
from src.data.data_sampler import BalancedSampler
import torchvision.transforms as transforms
import time
from tqdm import trange, tqdm
from src.models.model import get_model
import wandb
import pickle
import torch.nn as nn

#  ---------------  Training  ---------------
def train(
        model_name: str,
        batch_size: int = 12, lr=1e-4, weight_decay=0.9, epochs: int = 100, 
        experiment_name: str = None,
        checkpoint_every_epoch: int = 5,
        save_path: str = None,
        seed: int = 42,
    ):

    # ---------------------------- WandB ------------------------------ #
    wandb.init(
        # set the wandb project where this run will be logged
        project="deep_learning_in_cv",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": model_name,
        "dataset": "TACO",
        "epochs": epochs,
        },
        name=f"TACO_{model_name}_lr={lr}_epochs={epochs}_batch_size={batch_size}_seed={seed}",
    )

    # ---------------------------- Dataset ----------------------------- #
    # define transforms to resize - warping proposals
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # file obtained through create_crop_dataset.py
    path_train = r"/work3/s212725/WasteProject/data/json/train_region_proposals.json"
    
    # dataloader given json file
    train_dataset = RegionProposalsDataset(path_train, transform=transform)
    print(f"The length of the train dataset is of {len(train_dataset)}")

    # ------------------------ Balanced Sampler ------------------------ #
    # Load the labels obtained from \src\data\save_labels_to_pkl_file.py
    with open('../src/data/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    
    # Create the balanced sampler
    sampler = BalancedSampler(labels, pos_fraction=0.25, batch_size=32)

    #train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"The length of the train loader is of {len(train_loader)}")
    
    path_val = r"/work3/s212725/WasteProject/data/json/val_region_proposals.json"
    val_dataset = RegionProposalsDataset(path_val, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------------------------- Training --------------------------------- #
    # Set seed
    torch.manual_seed(seed)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Define the model, loss criterion and optimizer
    model, criterion, optimizer = get_model(model_name, lr=lr, weight_decay=weight_decay, device=device)
    criterion = nn.NLLLoss()
    
    print("CNN Architecture:")
    print(model)

    current_best_loss = torch.inf
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            model.train()
            for batch_idx, (images, labels, paths) in tqdm(enumerate(train_loader)):
                # Extract data
                images, labels = images.to(device), labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward + backward
                outputs = model(images)
                # Get predictions from log-softmax scores
                preds = torch.exp(outputs.detach()).topk(1)[1]
                # Get loss and compute gradient
                loss = criterion(outputs, labels)
                running_loss_train += loss.item()
                loss.backward()
                # Optimize
                optimizer.step()
                # Store accuracy
                print(preds)
                print(labels)
                train_acc = (preds.squeeze() == labels).sum().item()
                print(train_acc)
                running_acc_train += train_acc

            
            # Validation
            model.eval()
            # disable gradient propagation
            with torch.no_grad():
                for batch in tqdm(iter(val_loader)):
                    # Extract data                
                    images, labels, paths = batch
                    images, labels = images.to(device), labels.to(device)

                    # Forward + backward
                    outputs = model(images)
                    preds = torch.exp(outputs).topk(1)[1]

                    # Compute loss and accuracy
                    running_loss_val += criterion(outputs, labels) # For NLLLoss                    
                    correct = (preds.squeeze() == labels).sum().item() # count the number of correct predictions


            val_loss = running_loss_val / len(val_loader)
            if val_loss < current_best_loss and epoch % checkpoint_every_epoch == 0:
                current_best_loss = val_loss
                # Create and save checkpoint
                checkpoint = {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": {
                        'name': model_name,
                    },
                    "training_parameters": {
                        "save_path": save_path,
                        "lr": lr,
                        "optimizer": optimizer,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "device": device,
                    },
                    "data": {
                        "path": path_train,
                    },
                    "best_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                }
                os.makedirs(f"{save_path}/{experiment_name}", exist_ok=True)
                torch.save(checkpoint, f"{save_path}/{experiment_name}/best.ckpt")


            # Update progress bar
            train_loss_descr = (
                f"Train loss: {running_loss_train / len(train_loader):.3f}"
            )
            val_loss_descr = (
                f"Validation loss: {running_loss_val / len(val_loader):.3f}"
            )
            train_acc_descr = (
                f"Train accuracy: {running_acc_train / len(train_loader):.3f}"
            )
            val_acc_descr = (
                f"Validation accuracy: {running_acc_val / len(val_loader):.3f}"
            )
            t.set_description(
                f"EPOCH [{epoch + 1}/{epochs}] --> {train_loss_descr} | {val_loss_descr} | {train_acc_descr} | {val_acc_descr} | Progress: "
            )

            train_acc = running_acc_train / len(train_loader)
            train_loss = running_loss_train / len(train_loader)
            wandb.log({"Epoch":epoch, "train/acc": train_acc, "train/loss": train_loss})
            val_acc = running_acc_val / len(val_loader)
            wandb.log({"Epoch":epoch, "val/acc": val_acc, "val/loss": val_loss})

if __name__ == '__main__':

    # BASE_PATH = Path('projects/xai/XAI-ResponsibleAI')
    BASE_PATH = Path(r"/work3/s212725/WasteProject")
    
    save_path = BASE_PATH / 'models'
    model_name = 'ResNet18'
    lr = 1e-5
    wd = 1e-3
    batch_size = 32
    seed = 14
    experiment_name = f'{model_name}-lr{lr}.wd{wd}.bz{batch_size}.seed{seed}'

    train(
        model_name='ResNet18',
        batch_size=32,
        epochs=100,
        lr=lr,
        weight_decay=wd,
        experiment_name=experiment_name,
        checkpoint_every_epoch=5,
        save_path=save_path,
        seed=seed,
    )