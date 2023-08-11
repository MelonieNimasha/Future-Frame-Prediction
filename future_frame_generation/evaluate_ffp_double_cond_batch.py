from huggingface_hub import model_info
import inspect
from typing import List, Optional, Union

import numpy as np
import torch
import os
from skimage import io
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

import PIL
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler


image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def open_image(path, im_type = "RGB"):
  return PIL.Image.open(path).convert(im_type)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr_value = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr_value

def calculate_ssim(image_path1, image_path2):
    image1_read = io.imread(image_path1, as_gray=True)
    image2_read = io.imread(image_path2, as_gray=True)
    return ssim(image1_read, image2_read, data_range=image2_read.max() - image2_read.min())

# LoRA weights ~3 MB
model_path = "outsample7"

model_base = "runwayml/stable-diffusion-inpainting" 

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "tennis"
guidance_scale=0
num_samples = 4
generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

previous_frames = "data_prep/one_sample/previous_frames"
processed_frames = "data_prep/one_sample/processed_frames"
target_frames = "data_prep/one_sample/target_frames"
# previous_frames = "previous_frames"
# processed_frames = "processed_frames"
# target_frames = "target_frames"

image_paths = os.listdir(previous_frames) 
image2_paths = os.listdir(processed_frames)  
image3_paths = os.listdir(target_frames)  

# Initialize lists to store results
all_psnr_values = []
all_ssim_values = []

for i in range(len(image_paths)):
    image_path = os.path.join(previous_frames, image_paths[i])
    image2_path = os.path.join(processed_frames, image2_paths[i])
    image3_path = os.path.join(target_frames, image3_paths[i])

    image = open_image(image_path).resize((512, 512))
    image2 = open_image(image2_path, im_type = "L").resize((512, 512))
    image3 = open_image(image3_path).resize((512, 512))

    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=image2,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images


# Assuming you already have 'image', 'image2', 'image3', and 'images' defined

# Calculate PSNR between image3 and each image in the 'images' list
    psnr_values = []
    for generated_image in images:
        generated_np = np.array(generated_image)
        image3_np = np.array(image3)
        psnr_values.append(psnr(generated_np, image3_np))

    # Find the index of the image with maximum PSNR
    max_psnr_index = np.argmax(psnr_values)

    # Save the image with maximum PSNR
    if os.path.exists("data_prep/one_sample/generated_frames") and os.path.isdir("data_prep/test_small/generated_frames"):
        print()
    else:
        os.mkdir("data_prep/one_sample/generated_frames")
    out_path = f"data_prep/one_sample/generated_frames/ffp_doublecond_{i}.jpg"
    # out_path = f"generated_frames/ffp_doublecond_{i}.jpg"

    images[max_psnr_index].save(out_path)
    print("image ",out_path, " saved")

    # Evaluate the model on sample
    ssim_val = calculate_ssim(image3_path, out_path)
    psnr_val = max(psnr_values)

    all_psnr_values.append(psnr_val)
    all_ssim_values.append(ssim_val)

    print(psnr_val, ssim_val)

# Calculate the average SSIM and PSNR values
average_ssim = sum(all_ssim_values) / len(all_ssim_values)
average_psnr = sum(all_psnr_values) / len(all_psnr_values)

print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")