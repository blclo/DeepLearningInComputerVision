# Project 1.2: Object Detection RCNN - [TACO: Waste in the wild](http://tacodataset.org/)
In this project a deep learning object detection system that can automatically detect trash and litter and in images in the wild is built. This object detection can then be
deployed in robotic machines that can scan areas and collect and clean beaches, forests and roads.

1. Extracted object proposals for all the images of the dataset using Selecting
Search.
2. Finetune a convolutional neural network to classify object proposals.
3. Apply the model on the test images and implement non-maximum suppresion and Intersection over Union (IoU).
4. Evaluate the object detection performance using standard metrics

## Files Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── json_files         <- json data files used for model training. They are the results for the scripts in this codebase.
    ├── notebooks          <- useful Jupyter notebooks
    │   
    ├── scripts_region_proposal_dataset_creation
    │   ├── data_augmentation_of_gt_proposals.py  <- Script to perform data augmentation in the ground truth crops given its limited availability
    │   └── create_rp_dataset.py  <- Uses Selective Search algorithm and generates a dataset for crops and ground truth crops, 
    │                                  saving them into files and assigning their path to the label after computing Intersection Over Union.
    │   
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── split_dataset.py <- Splitting the json data into train, val and test files
    │   |   ├── download.py <- This script downloads TACO's images from Flickr given an annotation json file
    │   |   ├── rp_dataloader.py  <- Dataloader for the region proposals returning the images, their labels and paths
    │   │   ├── dataloader.py   <- Initial dataloader returning the images together with the data from the annotation file in the form a dictionary.
    │   |   └── data_sampler.py <- Given the biased dataset for background crops, the sampler ensures 75% of these per batch.
    │   │
    │   └── models         <- Scripts to train models
    │       │                 
    │       ├── model.py
    │       └── train_model.py
    │
    └── requirements.txt 
