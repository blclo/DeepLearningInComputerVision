# Project 1.2: Object Detection RCNN - [TACO: Waste in the wild](http://tacodataset.org/)
In this project a deep learning object detection system that can automatically detect trash and litter and in images in the wild is built. This object detection can then be
deployed in robotic machines that can scan areas and collect and clean beaches, forests and roads.

1. Extracted object proposals for all the images of the dataset using Selecting
Search.
2. Finetune a convolutional neural network to classify object proposals.
3. Apply the model on the test images and implement non-maximum suppresion and Intersection over Union (IoU).
4. Evaluate the object detection performance using standard metrics - Average Precision.

Results for the project are presented in the poster below:

![Poster P2](https://github.com/blclo/DeepLearningInComputerVision/raw/main/P1_2_WasteManagment/DLCV_CarolinaLopez-2.png)

### Length of datasets
- The length of the train dataset is 1050 images and 140638 region proposals, also known as crops.
- The length of the val dataset is 225 images and 29151 region proposals.
- The length of the test dataset is 225 images and 23224 region proposals.

### Results 
Results in training show an accuracy of 31.93% after including the augmented crops of the ground truth proposals - combating the dataset unbalance. Given the reduced amount of samples in the testing split, the obtained accuracy reaches 95.16%.

For the visual representation of the results, it was necessary to display the image together with its ground truth proposals, and the region proposals classified to have a positive label.

In order to do this a dictionary was included in the **create_rp_dataset.py** file. This dictionary contains the following:

- `test_dictionary["image_id"] = []` -> 225 images ids
- `test_dictionary["image_file_name"] = []` -> File names of the 225 images
- `test_dictionary["super_cats"] = []` -> An array of 225 arrays containing inside each the super categories of each images
- `test_dictionary["gt_bboxs"] = []` -> An array of 225 arrays. In each one, the ground truth bounding boxes for each image (out of the 225) are stored.
- `test_dictionary["rp_boxes"] = []` -> An array of 225 arrays, each of the 100 region proposal boxes per image
- `test_dictionary["crop_paths"] = []` -> An array of 225 arrays, each of 100 paths for each of the crops paths
- `test_dictionary["labels_of_crops_in_paths"] = []` -> 225 arrays, each of 100 labels for each of the crops per image

This dictionary is created and stored in testing_rp_dict.pkl. It will be later be loaded in the final notebook to print some results. 
In addition **test_model.py** is saving in **/data/json/test_loop_crops_paths_to_predicted_label.json** the path to the tested crops next to the predicted label.

### Learnings
Here's something to pay attention to. The bounding boxes for every crop are return of the form `(0, 1, 2, 3)`. However, this bounding box crops the image by doing: `image[crop[1]:crop[3], crop[0]:crop[2]]`

Thus, it is important to display the bounding boxes doing:
`rect = patches.Rectangle((x, y), w-x, h-y, linewidth=2, edgecolor='r', facecolor='none')`

Instead of doing:
`rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')`

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
    │   └── models         <- Scripts to train/test models
    │       │                 
    │       ├── model.py
    │       ├── test_model.py
    │       ├── utils.py
    │       └── train_model.py
    │
    └── requirements.txt 
