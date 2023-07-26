from huggingface_hub import model_info

# LoRA weights ~3 MB
model_path = "Melonie/pokemon-lora"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)   

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("Pink pokemon with barbie face", num_inference_steps=25).images[0]
image.save("pink_pokemon.png")



