###########################################################
# This script utilize `./dpo.py` to perform global schema linking with DPO
# Here's some paths/parameters need to be configured
#   1. sft_name: The filename that contains the training data
#   2. model_name: base LLM name or path
#   3. storage_dir: The base path to store the trained model
#   4. base_path: The path to the SFT-trained global linker
#   5. hyper-parameters: lr, beta, rpo_alpha
############################################################

#################### Constants ###################
export OMP_NUM_THREADS=4
#################### Directories ###################
model_name={MODEL_NAME}
storage_dir=/home/ubuntu

################### Change-able Path ###################
sft_name=${TRAINING_DATA_PATH}
base_path=${YOUR_PATH}

################### Tunable Parametera ##################
lr=$1  
beta=$2
rpo_alpha=$3

accelerate launch --config_file=/home/ubuntu/configs/TRAIN_DPO.json dpo.py \
    --model_storage_dir ${storage_dir}/tuned_Llama/linker-dpo-tuned \
    --finetune_data_dir ${sft_name} \
    --sft_base_path ${base_path} \
    --model_name_or_path ${model_name} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --save_strategy no \
    --output_dir ${storage_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --weight_decay 0.1 \
    --max_grad_norm 0.5 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --bf16 \
    --report_to none \
    --learning_rate ${lr} \
    --beta ${beta} \
    --rpo_alpha ${rpo_alpha} \
    --max_length 4500 \
    --max_prompt_length 4500 \
    --no_remove_unused_columns