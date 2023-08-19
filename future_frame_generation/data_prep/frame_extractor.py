import cv2
import PIL
from PIL import Image, ImageDraw
import os
import shutil
import sys
import numpy as np

def process_image(path):
  im = PIL.Image.open(path).convert("RGB")
  image = im.resize((512, 512))
  image.save(path)
  return

def list_files_in_folder(folder_path):
    return set(file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)))

def compare_and_remove_files(folder_paths):
    # Create sets to hold filenames for each folder
    folder_file_sets = [list_files_in_folder(path) for path in folder_paths]
    
    # Find the common filenames in all three folders
    common_files = folder_file_sets[0].intersection(*folder_file_sets[1:])
    
    # Remove files that are not present in the common_files set
    for i, folder_path in enumerate(folder_paths):
        files_to_remove = folder_file_sets[i] - common_files
        for file_to_remove in files_to_remove:
            file_path = os.path.join(folder_path, file_to_remove)
            os.remove(file_path)
            print(f"Removed: {file_path}")

def find_next_non_existing_integer(folder_path, base_filename):
    i = 0
    while True:
        filename = os.path.join(folder_path, f"{base_filename}_{i}.jpg")
        if not os.path.exists(filename):
            return i
        i += 1

def extract_frames(video_path, output_path_target, output_path_prev, output_path_prev_prev, saved_count=0, frame_rate=3):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame interval
    frame_interval = int(round(fps / frame_rate))
    
    # Initialize variables
    frame_count = 0
    start = 0
    
    # Iterate over the frames
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Check if a frame was successfully read
        if not ret:
            break
        
        # Increment the frame count
        frame_count += 1
        
        # Save the frame if it's at the desired interval to three folders of target image, previous image, previous image of previous
        if frame_count % frame_interval == 0:
            if start == 0:
                output_filename = f"{output_path_prev_prev}/frame_{saved_count}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)
                start +=1
            elif start ==1:
                output_filename = f"{output_path_prev_prev}/frame_{saved_count}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)
                output_filename = f"{output_path_prev}/frame_{saved_count-1}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)
                start +=1

            elif start ==2:
                output_filename = f"{output_path_prev_prev}/frame_{saved_count}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)
                output_filename = f"{output_path_prev}/frame_{saved_count-1}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)
                output_filename = f"{output_path_target}/frame_{saved_count-2}.jpg"
                cv2.imwrite(output_filename, frame)
                process_image(output_filename)

            saved_count += 1
    
    # Release the video file
    
    video.release()
    compare_and_remove_files([output_path_prev_prev, output_path_prev, output_path_target])
    return

def create_folder(folder_path):
    # Check if the folder already exists
    if os.path.exists(folder_path):
        print(f"Folder '{folder_path}' already exists.")
        # If it exists, remove it and recreate it
        try:
            shutil.rmtree(folder_path)
            print(f"Removed folder '{folder_path}'.")
        except OSError as e:
            print(f"Error removing folder '{folder_path}': {e}")
            return

    # Create the folder
    try:
        os.mkdir(folder_path)
        print(f"Created folder '{folder_path}'.")
    except OSError as e:
        print(f"Error creating folder '{folder_path}': {e}")

def process_videos_in_folder(video_folder_path, output_path_target, output_path_prev, output_path_prev_prev, frame_rate=3):

    #create new output files
    
        
    # Get a list of video file paths in the video folder
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.avi') or f.endswith('.mp4')]
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        n = find_next_non_existing_integer(output_path_target, "frame" )
        extract_frames(video_path, output_path_target, output_path_prev, output_path_prev_prev, frame_rate=frame_rate, saved_count = n)

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

def add_white_box(binary_image, padding=25):
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

def process_images(input_folder1, input_folder2, output_folder, output_folder2):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

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
        diff_image_bin_relaxed = diff_image_gray.point(lambda p: 0 if p < 25 else 255)
        diff_image_bin = diff_image_gray.point(lambda p: 0 if p < 112 else 255)

        #Prepare the relaxed mask
        relaxed_mask = add_white_box(diff_image_bin_relaxed)


        # Save the resulting binary image to the output folder with the same name
        output_path = os.path.join(output_folder, filename)
        output_path2 = os.path.join(output_folder2, filename)
        diff_image_bin.save(output_path)
        relaxed_mask.save(output_path2)
        

# Usage example

def create_data(video_folder_path):
    output_path_prev_prev = os.path.join(video_folder_path, "prev_previous_frames")
    output_path_prev = os.path.join(video_folder_path, "previous_frames")
    output_path_target = os.path.join(video_folder_path, "target_frames")
    output_processed = os.path.join(video_folder_path, "processed_frames")
    output_processed_relaxed = os.path.join(video_folder_path, "processed_frames_relaxed")
    for each in [output_path_target,output_path_prev, output_path_prev_prev, output_processed,output_processed_relaxed]:
            create_folder(each)

    process_videos_in_folder(video_folder_path, output_path_target, output_path_prev, output_path_prev_prev)
    process_images(output_path_prev_prev,output_path_prev,output_processed,output_processed_relaxed)
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py video_folder_path")
    else:
        video_folder_path = sys.argv[1]
        create_data(video_folder_path)