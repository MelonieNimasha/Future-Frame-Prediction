export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export HUB_MODEL_ID="inpaint-lora"
export DATA_PATH="video_frames"
export TEXT_TARGET="tennis"

accelerate launch --mixed_precision="fp16"  train_dreambooth_inpaint_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 --center_crop \
  --instance_data_dir=$DATA_PATH \
  --instance_prompt=$TEXT_TARGET \
  --class_prompt=$TEXT_TARGET \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --push_to_hub \
  --checkpointing_steps=50 \
  --hub_model_id=${HUB_MODEL_ID} \
  --seed=1337
