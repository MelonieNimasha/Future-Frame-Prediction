export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATA_PATH="/scratch/melonie/train_large/target_frames"
export COND_PATH="/scratch/melonie/train_large/previous_frames"
export COND_PATH2="/scratch/melonie/train_large/processed_frames"
export TEXT_TARGET="tennis"
export MODEL_OUT="models/outsample6_rate"

accelerate launch --mixed_precision="fp16"  ffp.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resolution=512 --center_crop \
  --instance_data_dir=$DATA_PATH \
  --cond_data_dir=$COND_PATH \
  --cond_data_dir2=$COND_PATH2 \
  --instance_prompt=$TEXT_TARGET \
  --class_prompt=$TEXT_TARGET \
  --train_batch_size=3 \
  --sample_batch_size=1 \
  --output_dir=$MODEL_OUT \
  --gradient_accumulation_steps=100 \
  --max_train_steps=100000\
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --checkpointing_steps=5000 \
  --seed=1337