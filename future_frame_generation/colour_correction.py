import PIL
import numpy as np
import torch
import os
from skimage import io
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

def colour_correction(input_path, output_path):
    processed = os.listdir(input_path)
    for i in range(len(previous)):
        name = f"frame_{i+start}.jpg"
        processed_path = os.path.join(processed_frames_relaxed, name)
        processed_image = open_image(processed_path, im_type = "L").resize((512, 512))
        im_out = replace_pixels(processed_image, previous_image, images[max_psnr_index])
        if os.path.exists("/scratch/melonie/val_large/corrected_frames"):
            print()
        else:
            os.mkdir("/scratch/melonie/val_large/corrected_frames")
        out_path = f"/scratch/melonie/val_large/corrected_frames/colour_corrected_{i}.jpg"
        im_out.save(out_path)

    