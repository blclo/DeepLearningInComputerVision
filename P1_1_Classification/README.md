# Project 1.1: Image Classification - HotDog/NoHotDog
- Design and train a CNN to do the classification task, evaluate its performance, and document the process.
- A dataset of images containing either hotdogs or something that is not a hotdog. The images come from the ImageNet categories:
pets, furniture, people, food, frankfurter, chili-dog, hotdog.

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
