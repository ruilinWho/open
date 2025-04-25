###########################################################
# This script utilize `./local.py` to perform local schema linking with SFT
# Here's some paths need to configure
#   1. {$1}: The filename that contains the training data
#      We used `data_for_training/schema_linking_all/local.json` to do this.
#   2. model: base LLM name or path
#   3. cache_dir: The cache directory that contains the base LLM
############################################################

export OMP_NUM_THREADS=4
# path
sh_path=$(dirname "$(readlink -f "$0")")
parent_dir=$(dirname "$sh_path")

# model
model=Qwen/Qwen2.5-Coder-3B-Instruct
lr=$1
cache_dir=/home/ubuntu/pretrained-models

accelerate launch --config_file=/home/ubuntu/configs/TRAIN_SFT_ZERO2.json local.py \
    --cache_dir ${cache_dir} \
    --model_storage_dir /home/ubuntu/tuned_models/local_linker/ \
    --finetune_data_dir bird_train.json \
    --validation_data_dir bird_dev.json \
    --model_name_or_path ${model} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --save_strategy no \
    --output_dir ${cache_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate ${lr} \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 30 \
    --bf16 \
    --report_to none \
    --max_seq_length 5000