# Project 1.1: Image Classification - HotDog/NoHotDog
- Design and train a CNN to do the classification task, evaluate its performance, and document the process.
- A dataset of images containing either hotdogs or something that is not a hotdog. The images come from the ImageNet categories:
pets, furniture, people, food, frankfurter, chili-dog, hotdog.

## Setup

Clone the repository and create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

## Files Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── __init__.py
    │   │   ├── dataloader.py
    │   |   └── hotdog-nothotdog <- The original data dump.
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── __init__.py
    │       ├── model.py
    │       └── train_model.py
    │
    └── requirements.txt 

## Setup
### Create environment
Run the following:

```
conda create -n deep_learning_in_cv python=3.10
```

Install the dependencies:
```
pip install -r requirements.txt
```

#### PyTorch - CPU
If running on CPU install Pytorch with the following command:

```
pip3 install torch torchvision torchaudio
```

#### PyTorch - GPU (CUDA 11.6)
If running on GPU with CUDA 11.6 install Pytorch with the following command:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Important concepts: LR, Momentum and Gradient
Gradient, learning rate, and momentum are all concepts related to optimization algorithms in machine learning. They play crucial roles in training models and updating the model parameters. Here's a brief explanation of each term and how they are connected:

1. Gradient:
In machine learning, the gradient refers to the vector of partial derivatives of a function with respect to its parameters. It represents the direction and magnitude of the steepest ascent or descent of the function. During model training, the gradient is computed to determine how the loss function changes with respect to the model parameters. The gradient guides the optimization process by indicating the direction in which the parameters should be adjusted to minimize the loss or maximize the objective function.

2. Learning Rate:
The learning rate is a hyperparameter that determines the step size at which the model parameters are updated during optimization. It controls the speed and convergence of the training process. A higher learning rate means larger parameter updates per iteration, which can lead to faster convergence but may risk overshooting the optimal solution. A lower learning rate provides smaller updates, which can improve stability but may result in slower convergence. The learning rate needs to be carefully chosen to strike a balance between convergence speed and accuracy.

3. Momentum:
Momentum is a technique used to accelerate the optimization process and overcome local minima. It introduces a "velocity" term that accumulates the gradients over time. The momentum term enhances the update of the model parameters by adding a fraction of the previous update vector. It allows the optimization algorithm to maintain its direction and speed, preventing it from getting stuck in flat regions or oscillating between narrow valleys. By incorporating momentum, the convergence can be faster, and the likelihood of getting trapped in suboptimal solutions can be reduced.

Connection:
The learning rate and momentum are hyperparameters that control how the optimization algorithm utilizes the gradient information for updating the model parameters. The learning rate determines the step size of parameter updates, while the momentum affects the direction and magnitude of the updates based on the accumulated gradients. Both the learning rate and momentum influence the optimization process, and finding appropriate values for these hyperparameters is crucial for effective model training.

The gradient provides the information about the slope and direction of the loss function with respect to the model parameters. It guides the optimization algorithm to update the parameters in a way that minimizes the loss function. The learning rate scales the gradient to determine the size of parameter updates, while the momentum affects the updates based on the historical gradients. Together, they contribute to the overall optimization process, influencing how the model parameters are adjusted during training to find an optimal solution.

### SGD (Stochastic Gradient Descent):
SGD is a popular optimization algorithm used to train machine learning models. It updates the model parameters based on the gradient of the loss function with respect to the parameters. In each iteration, SGD computes the gradient using a randomly selected subset of the training data (mini-batch) and updates the parameters in the direction of the negative gradient. Uses a fixed learning rate.

### RMSprop
Calculates a separate learning rate for each parameter by dividing the current gradient by the root mean square (RMS) of the previous gradients. 

### ADAM (Adaptive Moment Estimation):
ADAM is a separate optimization algorithm that combines elements from both RMSprop and momentum-based methods. It adapts the learning rate for each parameter based on the first and second moment estimates of the gradients. 

Both use Momentum. SGD uses momentum to incorporate information from previous gradients, allowing for smoother convergence and better escape from local minima. ADAM also employs similar momentum-like variables (first and second moment estimates), but the adaptation of these variables is performed automatically based on the gradients' statistics. The 2 momentum are:
- the first moment estimate (mean of gradients) 
- the second moment estimate (uncentered variance of gradients)

Regarding hyperparameter sensitivity: SGD requires careful tuning of the learning rate, which can be a challenging task. ADAM, on the other hand, is less sensitive to the initial learning rate and requires fewer hyperparameter adjustments.

