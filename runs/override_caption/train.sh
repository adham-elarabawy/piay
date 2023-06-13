export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="adhamelarabawy/islamic_art"

accelerate launch --mixed_precision="fp16"  ../../train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --output_dir="ckpt/" \
  --validation_prompt="islamic art"\
  --push_to_hub \
  --hub_model_id="adhamelarabawy/piay" \
  --report_to wandb \
  --enable_xformers_memory_efficient_attention \
  --noise_offset 0.1 \
  --train_batch_size 1 \
  --image_column "img" \
  --caption_column "caption" \
  --override_caption "islamic art"
  # --dataloader_num_workers=8 \
  # --tracker_project_name="piay" \
