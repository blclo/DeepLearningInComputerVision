# Project 2: Understanding Generative Adversarial Networks

## Files Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── images_poster      <- Images of the results
    ├── src                <- Source code for use in this project.
    │   └── model           
    │       └── train_model.py <- Script to train SVM to learn latent direction sunglasses/no_sunglasses 
    │    
    ├── Exercise_3.ipynb      <- Example notebook to understand GANs using MNIST
    │
    ├── GANS.ipynb   <- Main notebook containing interpolations, reconstructions...
    └── interpolation.py <- Script example of how the interpolation between two reconstructed images can be done


## Setting up your env
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

# StyleGAN 
The main component of the StyleGAN is known as the style-based generator. Here a latent input vector is fed into a mapping network. This network, composed of fully connected layers and non-linear activation functions outputs a intermediate latent space representation, also known as style space. Each dimension in this style-space representation `w` corresponds to an attribute or to an style element. 

This style vector is then fed into a synthesis-network which uses a technique called AdaIN : Adaptive Instance Normalization at each layer to model the normalization parameters (mean and standard deviation) based on the style-vector. These provides specific style-adjustments at different layers of the network. 

![model](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/model.png)

Due to the presense of artifacts in the activation maps, the model was improved by introducing StyleGAN2 which move the addition of bias and B (noise) outside the are of style. It also changed the instance normalization operation by the demodulation.

-  Demodulation involves dividing the activations of each feature map by their standard deviation. However, normalization normalizes the activations of each feature map by subtracting the mean and dividing by the standard deviation computed for each instance. 

The last improvement introduced, called StyleGAN2-ADA introduces Adaptive Discriminator Augmentation which means augmenting the real images during the discriminator training process. This enhances the training stability and generalization of the model. In addition to augmenting real images, StyleGAN2-ADA applies a technique called "mixing regularization." During training, instead of using only real images or only generated images, a random portion of the mini-batch is selected and replaced with either real or generated images. This mixing of real and generated images further enhances the generalization of the discriminator and helps it learn more robust decision boundaries.

## Generating images from a pretrained GAN model 
Make sure you have access to the pretrained network. In my case: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

Afterwards you can run - Ex: `python generate.py --outdir=out --trunc=1 --seeds=14,8,33,56 --network=ffql_style.pkl`

Results look like:

![generated_images](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/generated_random_faces.png)

## Reconstructing your own images
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

## Interpolate between two real images you reconstructed
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

## Applying some pre-trained latent directions to reconstructed images
Once the w latent vectors of our 'real' images have been obtained, we can always reconstruct them and apply some pre-trained latent directions such as gender, age or smile. This will be done by adding the latent vector to or original w with certain steps to increase the impact. The complete code can be found in the GANS.ipynb

- `w_interp =  w + aging_direction * step`

Some examples can be seen below:

![aging](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/gender_direction.png)
![gender](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/smile_direction.png)
![smile](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/aging_direction.png)

## Learning the latent directions
I have collected 20 images of people wearing sunglasses and 20 of people without. Afterwards, their latent representations have been learned using the projector.py script. Once done, it is possible to learn the latent direction training an SVM classifier. For this we can use the source code found in the `src` directory.

## Generating images from text prompt using CLIP
CLIP (Contrastive Language-Image Pretraining) is a neural network model that enables cross-modal understanding between images and text. CLIP is pretrained on a large corpus of publicly available image-text data from the internet. The model learns to encode both images and text into fixed-length representations using a combination of convolutional neural networks (CNNs) for images and transformer models for text.

Contrastive learning is a key aspect of CLIP. During pretraining, CLIP learns to align similar pairs of image-text examples while pushing apart dissimilar pairs. CLIP maps images and text into a shared joint embedding space, where semantically similar images and text are located closer to each other compared to dissimilar pairs. 

1. `!git clone https://github.com/vipermu/StyleCLIP`
2. `%cd StyleCLIP`
3. Install the requirements: 
`!pip install git+https://github.com/openai/CLIP.git`
`!pip install ftfy==5.8 opencv-python==4.5.1.48 regex==2020.11.13 torch==1.7.1 tqdm==4.56.0`
4. Load the `.pt` file (here)[https://github.com/vipermu/StyleCLIP/blob/master/README.md#stylegan-weights]
5. Run the prompt to generate: `!python clip_generate.py --prompt "The image of a blond kid with green eyes"`

Results below show the iterative process for the image generation:

![blond_kid](https://github.com/blclo/DeepLearningInComputerVision/blob/main/P2_Gans/images_poster/clip_generated_blond_kid.png)

The initialization of the latent code can have an impact on the generated images, although the significance may vary. Conditioning the generation on both an image and a text prompt is possible in some models, but it depends on the specific model's design and capabilities.

In StyleGAN2 ADA, the mapping network (also known as the style network) is responsible for mapping the initial z-space vectors to the intermediate w-space vectors. This mapping is learned during the training process. However, there is no explicit mapping function provided in the model to go back from w-space to z-space.

`z` is required to change the image initialization of the CLIP prompt and thus, inverse mapping must be performed. However, it's important to note that there is no direct, exact inverse mapping from w to z.
