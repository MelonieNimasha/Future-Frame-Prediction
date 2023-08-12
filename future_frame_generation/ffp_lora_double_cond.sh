export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export DATA_PATH="data_prep/one_sample/target_frames"
export COND_PATH="data_prep/one_sample/previous_frames"
export COND_PATH2="data_prep/one_sample/processed_frames"
export TEXT_TARGET="tennis"
export MODEL_OUT="models/outsample1"

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
  --output_dir=$MODEL_OUT \
  --gradient_accumulation_steps=1 \
  --max_train_steps=25000\
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --checkpointing_steps=5000 \
  --seed=1337