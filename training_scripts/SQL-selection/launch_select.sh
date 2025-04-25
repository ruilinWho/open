###########################################################
# This script utilize `./finetune_selector.py` to perform SQL stepwise selection training with SFT
# Here's some paths/parameters need to configure
#   1. sft_name: The filename that contains the training data
#      For Spider-dev, We used `data_for_training/SQL_selection/spider-selection.json` to do this.
#      For BIRD-dev, We used `data_for_training/SQL_selection/spider-selection.json` to do this.
#   2. model_name: base LLM name or path
#   3. cache_dir: The cache directory that contains the base LLM
#   4. lr($1) learning rate; epoch_num($2) training epoches (4 is suggested)
############################################################

export OMP_NUM_THREADS=4
# path
sh_path=$(dirname "$(readlink -f "$0")")
parent_dir=$(dirname "$sh_path")

config_dir=/home/ubuntu/configs
model_name=/home/ubuntu/pretrained-models/Llama-3.1-8B-Instruct
sft_name=${PATH_TO_SFT_DATA}
lr=$1
epoch_num=$2
cache_dir=/home/ubuntu/pretrained-models

accelerate launch --config_file=${config_dir}/TRAIN_SFT.json finetune_selector.py \
    --cache_dir ${cache_dir} \
    --model_storage_dir /home/ubuntu/tuned_llama/selection \
    --finetune_data_dir ${sft_name} \
    --model_name_or_path ${model_name} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --save_strategy no \
    --output_dir ${cache_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate ${lr} \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --num_train_epochs ${epoch_num} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 5 \
    --bf16 \
    --report_to none \
    --max_seq_length 13000 \
    --base_path ${BASE_MODEL_PATH}