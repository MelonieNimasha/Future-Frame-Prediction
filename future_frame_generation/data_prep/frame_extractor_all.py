import cv2
import PIL
from PIL import Image
import os
import shutil
import sys

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

def extract_frames(video_path, output_path_target, output_path_prev, output_path_prev_prev, saved_count=0, frame_rate=1):
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


# Usage example

def create_data(video_folder_path):
    output_path_prev_prev = os.path.join(video_folder_path, "prev_previous_frames")
    output_path_prev = os.path.join(video_folder_path, "previous_frames")
    output_path_target = os.path.join(video_folder_path, "target_frames")
    for each in [output_path_target,output_path_prev, output_path_prev_prev]:
            create_folder(each)

    process_videos_in_folder(video_folder_path, output_path_target, output_path_prev, output_path_prev_prev)
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py video_folder_path")
    else:
        video_folder_path = sys.argv[1]
        create_data(video_folder_path)