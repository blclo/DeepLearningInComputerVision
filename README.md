Special Course in Deep Learning in Computer Vision
==================================================

Repository for the Special Course in Deep Learning in Computer Vision @DTU

This repo will include 3 of the projects developed:
- P1_1: Image Classification: HotDog/NoHotDog
- P1_2: Object Detection: RCNN Waste Classification using the TACO dataset.
- P_2: Generative Adversarial Networks

## Accessing the DTU Cluster
1. Open your terminal and write:
```
ssh userid@login1.hpc.dtu.dk
```
2. Login with your credentials
3. Start up an interactive node using ```voltash```

## Setting up a venv
1. Create a directory in the HPC where you want your venvs to be stored
```mkdir venvs```
2. Check the available modules to load `module avail`
3. Load the python module:
```module load python3/3.10.7```
4. Create the venv:
```python3 -m venv NAME_VENV```
5. You can now activate your environment wherever using:
```source NAME_VENV/bin/activate```

## Setting up your Jupyter notebook
Make sure you have installed it in your env. If not do 
```
pip install notebook
```
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

## Submitting a job to the HPC DTU's Cluster
Once you have this text file (the job script, let us call it submit.sh), you must submit it by typing in a terminal the command (Source: https://www.hpc.dtu.dk/?page_id=1416):
```bsub < submit.sh```
Job Statistics:
```bstat```
