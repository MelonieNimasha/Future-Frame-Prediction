# Future Frame Generation with Stable Diffusion Model

This shows how to finetune different stable diffusion model checkpoints for different tasks using Low Rank Adaptation (LoRA). 
* Finetuning stable-diffusion-v1-5 for text_to_image generation and inferencing using finetuned model 
* Finetuning stable-diffusion-inpainting for image_to_image generation and inferencing using finetuned model 
* Finetuning stable-diffusion-inpainting for future frame generation and inferencing using finetuned model (in progress)

# Setup

conda create --name env_name
<br />
conda activate env_name
<br />
pip3 install diffusers["torch"] transformers
<br />
pip3 install diffusers["flax"] transformers
<br />
pip3 install accelerate
<br />
pip3 install git+https://github.com/huggingface/diffusers
<br />

git clone https://github.com/huggingface/diffusers.git
<br />
cd diffusers
<br />
pip3 install -e ".[torch]"
<br />
pip3 install -e ".[flax]"
<br />

huggingface-cli login 
(Enter token from https://huggingface.co/settings/tokens , create a token with write access)

## Text-to-Image 

### Finetune
cd example/text_to_image
<br />
(Move text_to_image/lora_test.sh file into the directory)
<br />
chmod +x lora_test.sh
<br />
./lora_test.sh

### Inference
(Move inpaint/ inference_text_to_image_lora.py file into the directory)
<br />
python3 inference_text_to_image_lora.py

## Inpaint

### Finetune
cd example/research_projects/dreambooth_inpaint
<br />
(Move inpaint/inpaint_lora_test.sh file and inpaint/video_frames into the directory)
<br />
chmod +x inpaint_lora_test.sh
<br />
./inpaint_lora_test.sh

### Inference
(Move inpaint/inference_inpaint_lora.py file into the directory)
<br />
python3 inference_inpaint_lora.py

## Future Frame Generation

### Data Preparation

pip3 install cv2
<br />
pip3 install pillow
<br />
cd Future_Frame_Generation/data_prep
<br />
copy the file path of video to "video_path" and existing video frame count in the dataset to saved_count
<br />
python3 frame_extractor.py
<br />
mkdir processed_images
<br />
python3 image_process.py

### Finetune

cd example/research_projects/dreambooth_inpaint
<br />
(Move future_frame_generation/ffp_lora_double_cond.sh and future_frame_generation/inference_ffp_double_cond.py file into the directory)
<br />
chmod +x ffp_lora_double_cond.sh
<br />
./ffp_lora_double_cond.sh


### Inference


## Reference 
https://github.com/huggingface/diffusers/tree/main
<br />
https://huggingface.co/blog/lora
<br />
https://huggingface.co/runwayml/stable-diffusion-inpainting













