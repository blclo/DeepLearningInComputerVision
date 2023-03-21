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
from src.models.model import Network

#  ---------------  Training  ---------------
def train(
        datafolder_path: str, datafile_name: str = 'catalan_dataset.pth',
        model_name: str = 'FullyConnected',
        batch_size: int = 128, 
        num_workers: int = 1, 
        test_proportion: float = 0.2, 
        val_proportion: float = 0.2, 
        split_type: str = 'random',
        lr=1e-3, epochs: int = 100, loss_type: str = 'BCE', optimizer: str = 'SGD', momentum: float = 0.9,
        experiment_name: str = str(int(round(time.time()))), save_path: str = '', 
        seed: int = 42,
    ):

    """
    Trains the model.

    Args:
        datafolder_path (str): _description_
        batch_size (int, optional): _description_. Defaults to 128.
        num_workers (int, optional): _description_. Defaults to 1.
        test_proportion (float, optional): _description_. Defaults to 0.2.
        val_proportion (float, optional): _description_. Defaults to 0.2.
        split_type (str, optional): _description_. Defaults to 'random'.
        lr (_type_, optional): _description_. Defaults to 1e-3.
        epochs (int, optional): _description_. Defaults to 100.
        experiment_name (str, optional): _description_. Defaults to str(int(round(time.time()))).
    """
    # Set seed
    set_seed(seed)
    current_best_loss = torch.inf

    # Load dataset
    dataset = CatalanJuvenileJustice(
        data_path=f"{datafolder_path}/processed/{datafile_name}"
    )

    columns = dataset.getColumns()

    # Split into training and test
    train_loader, val_loader, _ = dataset.get_loaders(
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1, 
        test_size=test_proportion, 
        val_size=val_proportion, 
        split_type=split_type,
    )

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Standardization constants
    mu      = train_loader.dataset.dataset.data.mean(axis=0).to(device)
    sigma   = train_loader.dataset.dataset.data.std(axis=0).to(device)

    # Define the model, loss criterion and optimizer
    model = get_model(model_name, dataset.n_attributes).to(device)
    criterion = get_loss_function(type=loss_type)
    optimizer = get_optimizer(model, type=optimizer, lr=lr, momentum=momentum)

    print("MLP Architecture:")
    print(model)

    fairness = Fairness_criteria(columns)
    Independence_dict = {}
    #print('Sensitive extended: ', sensitive_extended)
    #for i in sensitive_extended:
    #    Independence_dict[i] = 0
    writer = SummaryWriter(f"logs/{experiment_name}")
    with trange(epochs) as t:
        for epoch in t:
            running_loss_train, running_loss_val    = 0.0, 0.0
            running_acc_train,  running_acc_val     = 0.0, 0.0

            for batch in iter(train_loader):
                # Extract data                
                inputs, labels = batch['data'].to(device), batch['label'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Standardize inputs
                inputs = (inputs - mu) / sigma
                # Forward + backward
                outputs = model(inputs)
                y_pred = outputs['pred']

                loss = criterion(y_pred.float(), labels.float())
                running_loss_train += loss.item()
                loss.backward()

                # Optimize
                optimizer.step()
                
                # Store accuracy
                equals = (y_pred >= 0.5) == labels.view(*y_pred.shape)
                running_acc_train += torch.mean(equals.type(torch.FloatTensor))

            # Validation
            with torch.no_grad():
                for batch in iter(val_loader):
                    inputs_before, labels = batch['data'].to(device), batch['label'].to(device)
                    #print('input shape: ', inputs.shape)

                    # Standardize inputs
                    inputs = (inputs_before - mu) / sigma
                    
                    # Get predictions
                    outputs = model(inputs)
                    y_pred = outputs['pred']

                    # Compute loss and accuracy
                    running_loss_val += criterion(y_pred.float(), labels.float())
                    equals = (y_pred >= 0.5) == labels.view(*y_pred.shape)
                    running_acc_val += torch.mean(equals.type(torch.FloatTensor))

                    tmp_independence = fairness.Independence(y_pred, inputs_before)
                    Independence_dict_tmp = {k: (tmp_independence.get(k, 0) + Independence_dict.get(k, 0)) for k in set(tmp_independence) | set(Independence_dict)}
                    Independence_dict = Independence_dict_tmp
                    #print(Independence_dict)

            if running_loss_val / len(val_loader) < current_best_loss:
                current_best_loss = running_loss_val / len(val_loader)
                # Create and save checkpoint
                checkpoint = {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "model": {
                        'name': model_name,
                        'net': model.net,
                        'input_parameters': model.input_params,
                    },
                    "input_parameters": {
                        "input_size": dataset.n_attributes,
                        "output_size": 1,
                    },
                    "training_parameters": {
                        "save_path": save_path,
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "device": device,
                        "loss_type": loss_type,
                        "optimizer": {
                            "name": optimizer,
                            "momentum": momentum,
                        },
                    },
                    "data": {
                        "filename": datafile_name,
                        "standardization": {
                            "mu": mu,
                            "sigma": sigma,
                        },
                        "split": {
                            "test_proportion": test_proportion,
                            "val_proportion": val_proportion,
                            "split_type": split_type,
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

            writer.add_scalar(f'{loss_type}/train',         running_loss_train  / len(train_loader),    epoch)
            writer.add_scalar('accuracy/train',             running_acc_train   / len(train_loader),    epoch)
            writer.add_scalar(f'{loss_type}/validation',    running_loss_val    / len(val_loader),      epoch)
            writer.add_scalar('accuracy/validation',        running_acc_val     / len(val_loader),      epoch)

        print('\n-- Independence criteria --')
        for key, value in Independence_dict.items():
            print(key, 'has the \"acceptance\" rate of: ', value/(epochs*len(iter(val_loader))))


if __name__ == '__main__':
    # -----------------  Load dataset  -----------------
    trainset = Hotdog_NotHotdog(train=True)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = Hotdog_NotHotdog(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)
    
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

    # TODO: Define the loss function

    # TODO: Print model architecture
    
    # TODO: Notebook analysing the images

    # -----------------  Train model  -----------------
    # TODO: Define train function with basics
    train(
        datafolder_path = 'data',
        model_name='AutoEncoder',
        datafile_name='catalan_dataset_without_sensitives.pth',
        batch_size = 64, 
        epochs = 200, 
        lr=1e-4,
        test_proportion=0.4,
        loss_type='BCE',
        optimizer='Adam',
        experiment_name=f'AutoEncoder-0.4test_prop-no_sensitive_data.lr=1e-4',
        save_path='models',
    )

    # -----------------  Test model  -----------------
