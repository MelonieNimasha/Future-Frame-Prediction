from huggingface_hub import model_info
import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from diffusers import StableDiffusionInpaintPipeline

def open_image(path):
  return PIL.Image.open(path).convert("RGB")

# LoRA weights ~3 MB
model_path = "Melonie/inpaint-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)   

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionInpaintPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")


image = open_image('video_frames/frame_1.jpg').resize((512, 512))
prompt = "tennis"
guidance_scale=7.5
num_samples = 1
generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

images = pipe(
    prompt=prompt,
    image=image,
    mask_image=image,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images
images[0].save("inpaint_new.png")