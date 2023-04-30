#!/usr/bin/python
# 

import os
from pathlib2 import Path
import torch.nn.functional as F
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from src.data.rp_dataloader import RegionProposalsDataset
import time
from tqdm import trange, tqdm
from src.models.model import get_model
import wandb


#  ---------------  Training  ---------------
def train(
        model_name: str,
        batch_size: int = 12, lr=1e-4, weight_decay=0.9, epochs: int = 100, checkpoint_every_epoch: int = 5,
        seed: int = 42,
    ):

    #init wandb
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
    
    path_train = r"/work3/s212725/WasteProject/scripts_region_proposal_dataset_creation/all_train_region_proposals_together.json"
    train_dataset = RegionProposalsDataset(path_train, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    path_val = r"/work3/s212725/WasteProject/scripts_region_proposal_dataset_creation/all_val_region_proposals_together.json"
    val_dataset = RegionProposalsDataset(path_val, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set seed
    torch.manual_seed(seed)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"INFO - using device: {device}")

    # Define the model, loss criterion and optimizer
    model, criterion, optimizer, _ = get_model(model_name, lr=lr, weight_decay=weight_decay, device=device)
    
    print("CNN Architecture:")
    print(model)

    current_best_loss = torch.inf
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            model.train()
            for batch in tqdm(iter(train_loader)):
                # Extract data                
                inputs, labels, concepts = batch
                inputs, labels, concepts = inputs.to(device), labels.to(device), torch.stack(concepts).T.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward + backward
                outputs = model(inputs)
                # Get predictions from log-softmax scores
                preds = torch.exp(outputs.detach()).topk(1)[1]

                # Get loss and compute gradient
                loss = criterion(outputs, labels) # For NLLLoss                
                running_loss_train += loss.item()
                loss.backward()

                # Optimize
                optimizer.step()

                # Store accuracy
                equals = preds.flatten() == labels
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

            # Validation
            model.eval()
            with torch.no_grad():
                for batch in tqdm(iter(val_loader)):
                    # Extract data                
                    inputs, labels, concepts = batch
                    inputs, labels, concepts = inputs.to(device), labels.to(device), torch.stack(concepts).T.to(device)

                    # Forward + backward
                    outputs = model(inputs)
                    preds = torch.exp(outputs).topk(1)[1]

                    # Compute loss and accuracy
                    running_loss_val += criterion(outputs, labels) # For NLLLoss                    
                    equals = preds.flatten() == labels
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

            if running_loss_val / len(val_loader) < current_best_loss and epoch % checkpoint_every_epoch == 0:
                current_best_loss = running_loss_val / len(val_loader)
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
                        "path": processed_datafolder_path,
                        "normalization": {
                            "mu": list(normalization['mean'].numpy()),
                            "sigma": list(normalization['std'].numpy()),
                        },
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
            val_loss = running_loss_val / len(val_loader)
            wandb.log({"Epoch":epoch, "val/acc": val_acc, "val/loss": val_loss})

if __name__ == '__main__':

    # BASE_PATH = Path('projects/xai/XAI-ResponsibleAI')
    BASE_PATH = Path()

    raw_datafolder_path = BASE_PATH / 'data/raw/CUB_200_2011'
    processed_datafolder_path = BASE_PATH / 'data/processed/CUB_processed/class_attr_data_10'
    
    save_path = BASE_PATH / 'models'
    experiment_name = 'ResNet50-no_freeze-lr1e-5.wd1e-3.bz32.seed0'

    train(
        raw_datafolder_path=raw_datafolder_path,
        processed_datafolder_path=processed_datafolder_path,
        model_name='ResNet50',
        # from_checkpoint=save_path / experiment_name, # optional: for continuing training of models
        batch_size=32,
        epochs=100,
        lr=1e-5,
        weight_decay=1e-3,
        experiment_name=experiment_name,
        checkpoint_every_epoch=1,
        save_path=save_path,
        seed=0,
    )