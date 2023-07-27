import cv2
import PIL
from PIL import Image
import os

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

def extract_frames(video_path, output_path_target, output_path_prev, output_path_prev_prev, saved_count=0, frame_rate=0.5):
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


# Usage example
video_path = "v_tennis_01_07.avi"
output_path_prev_prev = "prev_previous_frames"
output_path_prev = "previous_frames"
output_path_target = "target_frames"
extract_frames(video_path, output_path_target,output_path_prev,output_path_prev_prev, frame_rate=2)
