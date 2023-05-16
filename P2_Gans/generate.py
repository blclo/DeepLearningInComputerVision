import dnnlib.tflib as tflib
import numpy as np
import PIL.Image

# Load pre-trained StyleGAN2 model
network_pkl = '/path/to/pretrained_model.pkl'
tflib.init_tf()
with open(network_pkl, 'rb') as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

# Generate random images
num_images = 5  # Number of images to generate
latent_size = Gs_network.input_shape[1]  # Size of the latent space

for i in range(num_images):
    # Generate random latent vector
    rnd = np.random.RandomState()
    latent_vector = rnd.randn(1, latent_size)

    # Generate image from the latent vector
    image = Gs_network.components.synthesis.run(latent_vector, randomize_noise=False, output_transform=dict(func=tflib.convert_images_to_uint8))

    # Convert the image to PIL format
    image_pil = PIL.Image.fromarray(image[0], 'RGB')

    # Display the generated image
    image_pil.show()
