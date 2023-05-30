import numpy as np
import matplotlib.pyplot as plt
import torch
import dnnlib
import torch.nn.functional as F
import pickle
import legacy
import os
import re
from typing import List
import click
import PIL.Image

device = torch.device('cuda')

# Load the pre-trained latent direction vectors
sunglasses_direction = np.load('latent_direction_sunglasses.npy')
sunglasses_direction = torch.from_numpy(sunglasses_direction).to(device)
print(sunglasses_direction.shape)

# Load your original latent vectors
original_latents = np.load('projected_w_celeb-2.npz')
w = original_latents['w']
w = torch.from_numpy(w).to(device)
print(w.shape)


network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
device = torch.device('cuda')

# Load the pre-trained generator model
with dnnlib.util.open_url(network_pkl) as f:
      G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

# Generate the reconstructed images
with torch.no_grad():
    img_original = G.synthesis(w, noise_mode='const')

num_steps = 10  # Number of interpolation steps
interpolated_images = []
interpolation_level = []
num_steps = [0, 25, 35, 45, 55, 75, 100]

# Perform linear interpolation
for step in num_steps:
    interpolation_level.append(step)
    w_interp =  w + sunglasses_direction * step
    with torch.no_grad():
        interp_img = G.synthesis(w_interp, noise_mode='const', force_fp32=True)
        interpolated_images.append(interp_img)

# Add the original images to the interpolation sequence
interpolated_images = interpolated_images 
interpolation_level = interpolation_level

# Create a grid to display the images
num_images = len(interpolated_images)
fig, ax = plt.subplots(1, num_images, figsize=(10, 2))

# Iterate over the images and plot them
for i in range(num_images):
    img = interpolated_images[i]
    img = img.squeeze().cpu().numpy()  # Convert the image tensor to a NumPy array
    img = (img + 1) / 2  # Rescale the pixel values from [-1, 1] to [0, 1]
    ax[i].imshow(img.transpose(1, 2, 0))
    ax[i].axis('off')
    ax[i].set_title(f"{interpolation_level[i]*100:.2f}%", fontsize=8)

plt.tight_layout()
plt.savefig("out/sample_celeb-2_sunglasses.png")
plt.show()
