export OMP_NUM_THREADS=4
# path
sh_path=$(dirname "$(readlink -f "$0")")
parent_dir=$(dirname "$sh_path")

# model
config_dir=/home/ubuntu/configs
model=/home/ubuntu/pretrained-models/Llama-3.1-8B-Instruct
lr=$1
epoch_num=$2
cache_dir=/home/ubuntu/pretrained-models

accelerate launch --config_file=${config_dir}/TRAIN_SFT.json finetune_selector.py \
    --cache_dir ${cache_dir} \
    --model_storage_dir /home/ubuntu/tuned_llama/spider-selection \
    --finetune_data_dir spider_deepseek_all_use_value-0311.json \
    --model_name_or_path ${model} \
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