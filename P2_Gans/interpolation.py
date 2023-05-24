import torch.nn.functional as F
import torch
import pickle
import dnnlib
import legacy
import os
import re
from typing import List

import click
import PIL.Image

# Load the .npz file
import numpy as np

# Load the .npz file
data_person1 = np.load('/content/stylegan2-ada-pytorch/out/aligned_clau/projected_w.npz')
# Load the .npz file
data_person2 = np.load('/content/stylegan2-ada-pytorch/out/projected_w.npz')

network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
device = torch.device('cuda')

# Load the pre-trained generator model
with dnnlib.util.open_url(network_pkl) as f:
      G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

w1 = data_person2['w']
w1 = torch.from_numpy(w1).to(device)
w2 = data_person1['w']
w2 = torch.from_numpy(w2).to(device)

# Generate the reconstructed images
with torch.no_grad():
    img1 = G.synthesis(w1, noise_mode='const')
    img2 = G.synthesis(w2, noise_mode='const')

num_steps = 10  # Number of interpolation steps
interpolated_images = []
interpolation_level = []
# Perform linear interpolation
for i in range(num_steps):
    alpha = i / (num_steps - 1)  # Interpolation factor between 0 and 1
    interpolation_level.append(alpha)
    w_interp = alpha * w1 + (1 - alpha) * w2
    with torch.no_grad():
        interp_img = G.synthesis(w_interp, noise_mode='const', force_fp32=True)
        interpolated_images.append(interp_img)

# Add the original images to the interpolation sequence
interpolated_images = interpolated_images 
interpolation_level = interpolation_level