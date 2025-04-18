# sh launch.sh 1e-5
export OMP_NUM_THREADS=4

# model
lr=$1

# Change-able data path
sft_name=DPO_SFT_Mix_1123.json

# Constants & Directories
# model_name=Qwen/Qwen2.5-Coder-7B-Instruct
model_name=meta-llama/Llama-3.1-8B-Instruct
dataset_dir=/home/ubuntu/dataset
storage_dir=/home/ubuntu
model_dir=/home/ubuntu/pretrained-models

cd /home/ubuntu/BIRD/schema-linking-7B/sft

accelerate launch --config_file=/home/ubuntu/configs/TRAIN_SFT.json tune_first_stage.py \
    --cache_dir ${model_dir} \
    --model_storage_dir ${storage_dir}/tuned_Llama/linker-dpo-base/ \
    --finetune_data_dir ${sft_name} \
    --validation_data_dir ${dataset_dir}/bird_dev_dynamic_link.json \
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
    --num_train_epochs 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --bf16 \
    --report_to none \
    --max_seq_length 13000 \
    --clean_long_data

accelerate launch --config_file=/home/ubuntu/configs/INFER.json validate_model.py \
    --model_storage_dir ${storage_dir}/tuned_Llama/linker-dpo-base/ \
    --model_name_or_path ${model_name} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --output_dir ${storage_dir} \
    --learning_rate ${lr} \
    --validation_data_dir ${dataset_dir}/bird_dev_dynamic_link.json