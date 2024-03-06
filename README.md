# Future Frame Generation with Stable Diffusion Model

This shows how to finetune stable-diffusion-inpainting checkpoint with LoRA for future frame generation and inferencing using finetuned model.

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

## Future Frame Generation

### Data Preparation

pip3 install cv2
<br />
pip3 install pillow
<br />
cd future_Frame_Generation/data_prep
<br />
python3 frame_extractor.py <path-to-folder_of_videos>



### Finetune

chmod +x ffp_lora_double_cond.sh
<br />
./ffp_lora_double_cond.sh


### Inference

python3 inference_ffp_double_cond.py


## Reference 
https://github.com/huggingface/diffusers/tree/main
<br />
https://huggingface.co/blog/lora
<br />
https://huggingface.co/runwayml/stable-diffusion-inpainting













