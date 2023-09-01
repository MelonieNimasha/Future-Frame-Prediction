export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATA_PATH="/scratch/melonie/train_large/target_frames"
export COND_PATH="/scratch/melonie/train_large/previous_frames"
export COND_PATH2="/scratch/melonie/train_large/processed_frames"
export DATA_PATH_VAL="/scratch/melonie/val_large/target_frames"
export COND_PATH_VAL="/scratch/melonie/val_large/previous_frames"
export COND_PATH2_VAL="/scratch/melonie/val_large/processed_frames"
export TEXT_TARGET="tennis"
export MODEL_IN="checkpoint-200000"
export MODEL_OUT="models/fully_trained"

accelerate launch --mixed_precision="fp16"  ffp.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resume_from_checkpoint=$MODEL_IN \
  --resolution=512 --center_crop \
  --instance_data_dir=$DATA_PATH \
  --cond_data_dir=$COND_PATH \
  --cond_data_dir2=$COND_PATH2 \
  --instance_data_dir_val=$DATA_PATH_VAL \
  --cond_data_dir_val=$COND_PATH_VAL \
  --cond_data_dir2_val=$COND_PATH2_VAL \
  --instance_prompt=$TEXT_TARGET \
  --train_batch_size=2 \
  --val_batch_size=1 \
  --sample_batch_size=1 \
  --output_dir=$MODEL_OUT \
  --gradient_accumulation_steps=2 \
  --max_train_steps=300000\
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --checkpointing_steps=10000 \
  --seed=1337

