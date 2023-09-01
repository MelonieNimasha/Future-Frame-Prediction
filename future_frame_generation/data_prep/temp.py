import os
import shutil

# Define source, destination for odd, and destination for even folders
source_folder = '/scratch/melonie/val_large/target_frames'
odd_folder = '/scratch/melonie/test_large/target_frames'
even_folder = '/scratch/melonie/val_new/target_frames'

# Create the destination folders if they don't exist
for folder in [odd_folder, even_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# List files in the source folder
files = os.listdir(source_folder)

# Iterate through the files and move them to the appropriate folders
for i in range(len(files)):
    file = f"frame_{i}.jpg"
    if i % 2 == 0:
        j = int(i / 2)
        new_name = f"frame_{j}.jpg"
        destination = os.path.join(even_folder, new_name)
    else:
        destination = os.path.join(odd_folder, new_name)
        
    source_path = os.path.join(source_folder, file)
    shutil.move(source_path, destination)
    print(f"Moved '{file}' to '{destination}'")
