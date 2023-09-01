# Future Frame Prediction with Stable Diffusion Model

This shows how to Finetune stable-diffusion-inpainting checkpoint for future frame prediction and inferencing using finetuned model.

# Setup

- conda create --name env_name
- conda activate env_name
<br />
pip3 install diffusers["torch"] transformers
<br />
pip3 install diffusers["flax"] transformers
<br />
pip3 install accelerate
<br />
pip3 install git+https://github.com/huggingface/diffusers
<br />

## Future Frame Generation

### Data Preparation for training and inference

<After frame extraction from source videos, the dataset will have 3 main folders as prev_prevous_frames, previous_frames, and  target_frames, containing corresponding consecutive 3 frames of each data sample. Another 2 folders are created, named processed_frames and processed_frames_relaxed_cleaned, containg the corresponding binary masks for FFP-LDM and the other binary masks for background correction>

data source: tennis swing videos from https://www.crcv.ucf.edu/research/data-sets/ucf-youtube-action/ 


pip3 install cv2
<br />
pip3 install pillow
<br />
cd future_Frame_Generation/data_prep
<br />
python3 frame_extractor.py <path-to-folder_of_videos>
<br />
<Update input_folder1, input_folder2 and output_folder2 in the mask2_creator.py with folders path to required prev_prevous_frames folder, previous_frames folder and the output folder to contain background correction mask>
<br />
python3 mask2_creator.py



### Finetune

<Update the data_path, cond_path, cond_path2, data_path_val, cond_path_val, and cond_path2_val in ffp_lora_double_cond.sh file with target_frames, previous_frames, processed_frames folders of training and vaidation datasets>
<br />
chmod +x ffp_lora_double_cond.sh
<br />
./ffp_lora_double_cond.sh


### Evaluate

<Update the previous_frames, target_frames, processed_frames, and processed_frames_relaxed_cleaned folders of test data in the evaluate_ffp_double_cond_batch.py file>
<br />
python3 evaluate_ffp_double_cond_batch.py

### Evaluate Baseline

<Update the previous_frames and target_frames folders of test data in the evaluate_baseline.py>
<br />
python3 evaluate_baseline.py




## Reference 
https://github.com/huggingface/diffusers/tree/main
<br />
https://huggingface.co/blog/lora
<br />
https://huggingface.co/runwayml/stable-diffusion-inpainting













