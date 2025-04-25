###########################################################
# This script utilize `./tune_generation.py` to perform SQL generation training with SFT
# Here's some paths need to configure
#   1. sft_name: The filename that contains the training data
#      We used `data_for_training/SQL_generation/Spider-BIRD-Gen-Mix.json` to do this.
#   2. model_name: base LLM name or path
#   3. storage_dir: The base path to store the trained model
#   4. model_dir: The cache directory that contains the base LLM
############################################################
export OMP_NUM_THREADS=4

# model
lr=$1

sft_name=${PATH_TO_SFT_DATA}

# Constants & Directories
model_name=${YOUR_MODEL}
storage_dir=/home/ubuntu/
model_dir=/home/ubuntu/pretrained-models

accelerate launch --config_file=/home/ubuntu/configs/TRAIN_SFT.json tune_generation.py \
    --cache_dir ${model_dir} \
    --model_storage_dir ${storage_dir}/tuned_models/ \
    --finetune_data_dir ${sft_name} \
    --model_name_or_path ${model_name} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --save_strategy no \
    --output_dir ${storage_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate ${lr} \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --bf16 \
    --report_to none \
    --max_seq_length 13000 \
    --clean_long_data