# Project 2: Understanding Generative Adversarial Networks
### Setting up your env
You environment needs to match the required documentation from the https://github.com/NVlabs/stylegan2-ada-pytorch repo.
1. It is important to load `Python 3.7`, in the HPC you can do:
- Run `module avail` to see the available Python versions to load
- In our case we loaded `module load python3/3.7.14`
2. Create the venv doing `python3 -m venv NAME_VENV`
3. Activate it `source NAME_VENV/bin/activate`
4. Install the required PyTorch version `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html`
5. Install the required packages: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`
6. In my case I also had to install a specific version of urllib `pip install urllib3==1.26.6`. See https://github.com/NVlabs/stylegan2-ada-pytorch/issues/39 for more info.
7. Everything should be in place to enjoy creating!

### Generating images from a pretrained GAN model 
Make sure you have access to the pretrained network. In my case: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

Afterwards you can run - Ex: ´python generate.py --outdir=out --trunc=1 --seeds=14,8,33,56 --network=ffql_style.pkl´

Results look like:
![generated_images](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/generated_random_faces.png)

### Reconstructing your own images
To "reconstruct your own images" in the context of StyleGAN2 refers to the process of finding the latent codes or latent vectors that can generate images that closely resemble your input images. In other words, it's about finding the latent representation that, when passed through the generator network of StyleGAN2, produces images that are similar to the provided input images.

To do this, you can start by aligning your images using https://github.com/happy-jihye/FFHQ-Alignment/tree/master/FFHQ-Alignmnet

1. Clone the repo
2. Run `pip install face-alignment`
3. Get your images in JPG file format in the raw_images directory
3. Run `python ffhq-align.py -s raw_images/ -d aligned_images/`

![aligned_images](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/aligned_image_example.png)

Afterwards we will make use of the projector.py script.
1. In order to run this script succesfully it is required to run:
`pip install --upgrade imageio-ffmpeg`
2. Run `!python projector.py --outdir=out --target=align-carol.png --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl`

![projected_images](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/result_projector.png)

### Interpolate between two real images you reconstructed
To interpolate between two real images that you have reconstructed using StyleGAN2, you can follow these steps:

1. Reconstruct the latent codes: Use the "projector.py" script or any other method to obtain the latent codes for the two real images you want to interpolate between. This will give you the latent vectors that represent the respective images in the latent space.

2. Choose interpolation points: Select a set of interpolation points between 0 and 1. These points will determine the extent of interpolation between the two latent codes. For example, you can choose interpolation points such as [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] to create six intermediate images.

3. Perform interpolation: For each interpolation point, compute the linear interpolation between the two latent codes. Linear interpolation can be performed by taking a weighted average of the two latent codes using the interpolation point as the weight. Mathematically, the interpolated latent code `z_interpolated` at a given interpolation point `alpha` can be calculated as:
   ```
   z_interpolated = (1 - alpha) * z1 + alpha * z2
   ```
   where `z1` and `z2` are the latent codes of the two original images.

4. Generate images: Pass each interpolated latent code `z_interpolated` through the generator network of StyleGAN2 to generate the corresponding interpolated images.

5. Visualize the results: Display the original two images, along with the interpolated images, in a sequence to observe the smooth transition between them.

By interpolating between the latent codes of two reconstructed real images, you can create a sequence of images that smoothly transitions between the two originals. This interpolation technique allows you to explore and visualize the latent space of the generative model and observe the changes in image features and characteristics along the interpolation path. Here's an example of the result:

![interpolated_images](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/interpolation.png)

