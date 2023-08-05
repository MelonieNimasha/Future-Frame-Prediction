export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export HUB_MODEL_ID="ffp-double-cond-lora"
export DATA_PATH="data_prep/test_small/target_frames"
export COND_PATH="data_prep/test_small/previous_frames"
export COND_PATH2="data_prep/test_small/processed_frames"
export TEXT_TARGET="tennis"

accelerate launch --mixed_precision="fp16"  ffp.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 --center_crop \
  --instance_data_dir=$DATA_PATH \
  --cond_data_dir=$COND_PATH \
  --cond_data_dir2=$COND_PATH2 \
  --instance_prompt=$TEXT_TARGET \
  --class_prompt=$TEXT_TARGET \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --push_to_hub \
  --checkpointing_steps=50 \
  --hub_model_id=${HUB_MODEL_ID} \
  --seed=1337
