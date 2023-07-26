# Future-Frame-Prediction

This shows how to finetune different stable diffusion model checkpoints for different tasks. 
* Finetuning stable-diffusion-v1-5 for text_to_image generation and inferencing using finetuned model ✅
* Finetuning stable-diffusion-inpainting for image_to_image generation and inferencing using finetuned model ✅
* Finetuning stable-diffusion-inpainting for future frame generation and inferencing using finetuned model (in progress)

# Guildlines

conda create --name <env name>
conda activate <env name>
pip3 install diffusers["torch"] transformers
pip3 install diffusers["flax"] transformers
pip3 install accelerate
pip3 install git+https://github.com/huggingface/diffusers

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip3 install -e ".[torch]"
pip3 install -e ".[flax]"

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












