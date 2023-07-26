# Future-Frame-Prediction

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
(Move text_to_image/lora_test.sh into the directory)
chmod +x lora_test.sh
./lora_test.sh

### Inference
python3 inference_text_to_image_lora.py













