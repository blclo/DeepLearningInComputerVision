Special Course in Deep Learning in Computer Vision
==================================================

Repository for the Special Course in Deep Learning in Computer Vision @DTU

## Setup

Clone the repository and create a virtual environment (with Python 3.10). A pre-defined environment running with CUDA 11.6 can be created like:

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

## Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── catalan_data_course            <- The original, immutable data dump.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   |   ├── __init__.py
    │   │   └── dataloader.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── __init__.py
    │       ├── model.py
    │       └── train_model.py
    │
    └── requirements.txt 

## Accessing the DTU Cluster
1. Open your terminal and write:
```
ssh userid@login1.hpc.dtu.dk
```
2. Login with your credentials
3. Activate your Conda env
4. Start up an interactive node using ```voltash```

## Setting up your jupyter notebook
1. Start it inside your interactive node
```
jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME
```
2. Copy the values in your url that look like this:
```
n-62-20-1:40001
```
3. Open another terminal and write. Please note substitute USER by your username and make sure the above line is included after L8080:
```
ssh USER@l1.hpc.dtu.dk -g -L8080:n-62-20-1:40001 -N
```
4. Enter your credentials
5. It will look as if nothing happened, but open your browser and write ```http://localhost:8080/tree?```

## Setting up a venv
1. Create a directory in the HPC where you want your venvs to be stored
```mkdir venvs```
2. Load the python module:
```module load python3/3.10.7```
3. Create the venv:
```python3 -m venv NAME_VENV```
4. You can now activate your environment wherever using:
```source NAME_VENV/bin/activate```