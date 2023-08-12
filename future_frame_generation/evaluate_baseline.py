import os
from PIL import Image
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim

def psnr(image1_path, image2_path):
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr_value = 20 * np.log10(pixel_max / np.sqrt(mse))
    return psnr_value

def calculate_psnr_for_matching_filenames(folder1, folder2):
    psnr_list = []

    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    common_files = set(folder1_files) & set(folder2_files)

    for filename in common_files:
        image1_path = os.path.join(folder1, filename)
        image2_path = os.path.join(folder2, filename)
        psnr_value = psnr(image1_path, image2_path)
        psnr_list.append((filename, psnr_value))

    return psnr_list

def calculate_average_psnr(folder1, folder2):
    psnr_list = calculate_psnr_for_matching_filenames(folder1, folder2)
    psnr_values = [psnr_value for _, psnr_value in psnr_list]
    average_psnr = sum(psnr_values) / len(psnr_values)
    return average_psnr

def calculate_ssim(image_path1, image_path2):
    image1_read = io.imread(image_path1, as_gray=True)
    image2_read = io.imread(image_path2, as_gray=True)
    return ssim(image1_read, image2_read, data_range=image2_read.max() - image2_read.min())

def calculate_ssim_between_folders(folder1, folder2):
    ssim_scores = []
    for filename in os.listdir(folder1):
        if filename in os.listdir(folder2):
            image_path1 = os.path.join(folder1, filename)
            image_path2 = os.path.join(folder2, filename)
            score = calculate_ssim(image_path1, image_path2)
            ssim_scores.append((filename, score))
    return ssim_scores

def calculate_average_ssim(folder1, folder2):
    ssim_scores = calculate_ssim_between_folders(folder1, folder2)
    ssim_values = [ssim_value for _, ssim_value in ssim_scores]
    average_ssim = sum(ssim_values) / len(ssim_values)
    return average_ssim

# Example usage:
folder1_path = "data_prep/one_sample/previous_frames"
folder2_path = "data_prep/one_sample/target_frames"
psnr_results = calculate_average_psnr(folder1_path, folder2_path)
print("PSNR", psnr_results)

ssim_results = calculate_average_ssim(folder1_path, folder2_path)
print("SSIM", ssim_results)



