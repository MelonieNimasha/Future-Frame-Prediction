from huggingface_hub import model_info
import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

def open_image(path):
  return PIL.Image.open(path).convert("RGB")

# LoRA weights ~3 MB
model_path = "outsample1"

model_base = "runwayml/stable-diffusion-inpainting"  

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


image = open_image('data_prep/test_small/previous_frames/frame_0.jpg').resize((512, 512))
image2 = open_image('data_prep/test_small/processed_frames/frame_0.jpg').resize((512, 512))
prompt = "tennis"
guidance_scale=0
num_samples = 4
generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

images = pipe(
    prompt=prompt,
    image=image,
    mask_image=image2,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images
images[0].save("data_prep/test_small/generated_frames/frame_0_0.png")
images[1].save("data_prep/test_small/generated_frames/frame_0_1.png")
images[2].save("data_prep/test_small/generated_frames/frame_0_2.png")
images[3].save("data_prep/test_small/generated_frames/frame_0_3.png")