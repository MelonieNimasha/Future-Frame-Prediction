import cv2

def extract_frames(video_path, output_path, frame_rate=0.5):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame interval
    frame_interval = int(round(fps / frame_rate))
    
    # Initialize variables
    frame_count = 0
    saved_count = 0
    
    # Iterate over the frames
    while True:
        # Read the next frame
        ret, frame = video.read()
        
        # Check if a frame was successfully read
        if not ret:
            break
        
        # Increment the frame count
        frame_count += 1
        
        # Save the frame if it's at the desired interval
        if frame_count % frame_interval == 0:
            output_filename = f"{output_path}/frame_{saved_count}.jpg"
            cv2.imwrite(output_filename, frame)
            saved_count += 1
    
    # Release the video file
    video.release()


# Usage example
video_path = "v_walk_dog_01_01.avi"
output_path = "vf"
extract_frames(video_path, output_path, frame_rate=2)
