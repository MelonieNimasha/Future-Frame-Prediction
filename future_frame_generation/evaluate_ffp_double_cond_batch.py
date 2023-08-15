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

def replace_pixels(new_image, image1, image2):
    if image1.size != image2.size or new_image.size != image1.size:
        raise ValueError("Image sizes must match.")

    new_pixels = new_image.load()
    pixels1 = image1.load()
    pixels2 = image2.load()

    result_image = PIL.Image.new('RGB', image1.size)

    for x in range(image1.width):
        for y in range(image1.height):
            if new_pixels[x, y] == 0:
                result_image.putpixel((x, y), pixels1[x, y])
            else:
                result_image.putpixel((x, y), pixels2[x, y])

    return result_image

# LoRA weights ~3 MB
model_path = "models/outsample_test"

model_base = "runwayml/stable-diffusion-inpainting" 

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "tennis"
guidance_scale=0
num_samples = 4
generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

previous_frames = "data_prep/one_sample_test/previous_frames"
target_frames = "data_prep/one_sample_test/target_frames"
processed_frames = "data_prep/one_sample_test/processed_frames"
processed_frames_relaxed = "data_prep/one_sample_test/processed_frames_relaxed"
# previous_frames = "previous_frames"
# processed_frames = "processed_frames"
# target_frames = "target_frames"

previous = os.listdir(previous_frames) 
target = os.listdir(target_frames)  
process = os.listdir(processed_frames)  
processed = os.listdir(processed_frames_relaxed) 

# Initialize lists to store results
all_psnr_values = []
all_ssim_values = []
start = 0

for i in range(len(previous)):
    name = f"frame_{i+start}.jpg"
    prev_path = os.path.join(previous_frames, name)
    target_path = os.path.join(target_frames, name)
    process_path = os.path.join(processed_frames, name)
    processed_path = os.path.join(processed_frames_relaxed, name)


    previous_image = open_image(prev_path).resize((512, 512))
    target_image = open_image(target_path).resize((512, 512))
    process_image = open_image(process_path, im_type = "L").resize((512, 512))
    processed_image = open_image(processed_path, im_type = "L").resize((512, 512))
    

    images = pipe(
        prompt=prompt,
        image=previous_image,
        mask_image=process_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images


# Assuming you already have 'image', 'image2', 'image3', and 'images' defined

# Calculate PSNR between image3 and each image in the 'images' list
    psnr_values = []
    for generated_image in images:
        generated_np = np.array(generated_image)
        target_np = np.array(target_image)
        psnr_values.append(psnr(generated_np, target_np))

    # Find the index of the image with maximum PSNR
    max_psnr_index = np.argmax(psnr_values)

    # Save the image with maximum PSNR
    if os.path.exists("data_prep/one_sample_test/generated_frames") and os.path.isdir("data_prep/one_sample_test/generated_frames"):
        print()
    else:
        os.mkdir("data_prep/one_sample_test/generated_frames")
    out_path = f"data_prep/one_sample_test/generated_frames/ffp_doublecond_{i+start}.jpg"
    # out_path = f"generated_frames/ffp_doublecond_{i}.jpg"
    
    im_out = replace_pixels(processed_image, previous_image, images[max_psnr_index])
    generated_np_new = np.array(im_out)
    im_out.save(out_path)
    print("image ",out_path, " saved")

    # Evaluate the model on sample
    ssim_val = calculate_ssim(target_path, out_path)
    psnr_val = psnr(target_np, generated_np_new)

    all_psnr_values.append(psnr_val)
    all_ssim_values.append(ssim_val)

    print(psnr_val, ssim_val)

# Calculate the average SSIM and PSNR values
average_ssim = sum(all_ssim_values) / len(all_ssim_values)
average_psnr = sum(all_psnr_values) / len(all_psnr_values)

print(f"Average PSNR: {average_psnr}")
print(f"Average SSIM: {average_ssim}")