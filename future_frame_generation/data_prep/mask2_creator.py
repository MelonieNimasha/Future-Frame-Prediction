import cv2
import PIL
from PIL import Image, ImageDraw
import os
import shutil
import sys
import numpy as np
from skimage.measure import label, regionprops

def pixel_wise_difference(image1_path, image2_path):
    # Open the two images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Ensure the images have the same size
    if image1.size != image2.size:
        raise ValueError("Images must have the same size.")

    # Get the pixel data for both images
    pixels1 = image1.load()
    pixels2 = image2.load()

    # Create a new image to store the pixel-wise difference
    diff_image = Image.new('RGB', image1.size)
    pixels_diff = diff_image.load()

    # Calculate the pixel-wise difference
    for x in range(image1.width):
        for y in range(image1.height):
            r_diff = abs(pixels1[x, y][0] - pixels2[x, y][0])
            g_diff = abs(pixels1[x, y][1] - pixels2[x, y][1])
            b_diff = abs(pixels1[x, y][2] - pixels2[x, y][2])
            pixels_diff[x, y] = (r_diff, g_diff, b_diff)

    return diff_image

def larget_connected_component(binary_mask):
    binary_mask_array = np.array(binary_mask)

    # Label connected components
    labeled_mask = label(binary_mask_array)

    # Calculate properties of connected components
    props = regionprops(labeled_mask)

    # Find the largest connected component
    largest_component = max(props, key=lambda prop: prop.area)

    # Create a new binary mask containing only the largest connected component
    largest_component_mask = np.zeros_like(binary_mask_array)
    largest_component_mask[labeled_mask == largest_component.label] = 225

    # Save or use the largest connected component mask as needed
    largest_component_image = Image.fromarray(largest_component_mask)
    return largest_component_image


def add_white_box(binary_image, padding=50):
    # Find the bounding box coordinates around white pixels
    non_zero_coords = [(x, y) for x in range(binary_image.width) for y in range(binary_image.height) if binary_image.getpixel((x, y)) > 0]
    if not non_zero_coords:
        return binary_image

    min_x = max(0, min(non_zero_coords, key=lambda item: item[0])[0] - padding)
    min_y = max(0, min(non_zero_coords, key=lambda item: item[1])[1] - padding)
    max_x = min(binary_image.width - 1, max(non_zero_coords, key=lambda item: item[0])[0] + padding)
    max_y = min(binary_image.height - 1, max(non_zero_coords, key=lambda item: item[1])[1] + padding)

    # Create a new image with a black background
    new_image = Image.new('L', binary_image.size, color=0)

    # Draw a white bounding box on the new image
    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=255, fill=255)

    return new_image

def process_images(input_folder1, input_folder2, output_folder2):
    # Ensure the output folder exists
    os.makedirs(output_folder2, exist_ok=True)

    # Get a list of image filenames from the first input folder
    image_filenames = os.listdir(input_folder1)

    for filename in image_filenames:
        # Build the full paths to the images in both input folders
        image1_path = os.path.join(input_folder1, filename)
        image2_path = os.path.join(input_folder2, filename)

        # Perform pixel-wise difference and get the resulting image
        diff_image = pixel_wise_difference(image1_path, image2_path)

        # Convert the difference image to grayscale
        diff_image_gray = diff_image.convert('L')  # 'L' mode for grayscale

        # Apply a binary threshold to create a binary image
        # You can adjust this threshold value as needed
        diff_image_bin_relaxed = diff_image_gray.point(lambda p: 0 if p < 50 else 255)
        #Get the largets connected componenet of the mask
        diff_image_bin_cleaned = larget_connected_component(diff_image_bin_relaxed)
        diff_image_bin_relaxed.save("diff_image_bin_relaxed.jpg")
        diff_image_bin_cleaned.save("diff_image_bin_cleaned.jpg")
        #Prepare the relaxed mask
        relaxed_mask = add_white_box(diff_image_bin_cleaned)
        # Save the resulting binary image to the output folder with the same name
        output_path2 = os.path.join(output_folder2, filename)
        relaxed_mask.save(output_path2)



input_folder1 = "/scratch/melonie/test_large_new/prev_previous_frames"
input_folder2 = "/scratch/melonie/test_large_new/previous_frames"
output_folder2 = "/scratch/melonie/test_large_new/processed_frames_relaxed_cleaned"

process_images(input_folder1, input_folder2, output_folder2)