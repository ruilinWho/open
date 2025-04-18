export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=36000

GLOBAL_LINK_PATH=$1
GENERATOR_MODEL_PATH=$2
SELECT_MODEL_PATH=$3
LOCAL_MODEL_PATH=$4

################################### Step(0) Global_Link: ######################
accelerate launch --config_file=/home/ubuntu/configs/INFER.json step0_global.py \
    --global_model_path ${GLOBAL_LINK_PATH} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16

################################## Step(1) Local_Link: #########################################
accelerate launch --config_file=/home/ubuntu/configs/INFER.json step1_refine_sql.py \
    --refine_model_path ${LOCAL_MODEL_PATH} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --persistent_input_path pipeline/0_global_link.json

#################################### Step(2) Generate SQL Candidates: #########################################
accelerate launch --config_file=/home/ubuntu/configs/INFER.json step2_generate.py \
    --model_storage_dir ${GENERATOR_MODEL_PATH} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --refined_input_path pipeline/1_refined_input.json


#################################### Step(3) Select SQL Output: #########################################
accelerate launch --config_file=/home/ubuntu/configs/INFER.json step3_select.py \
    --model_storage_dir ${SELECT_MODEL_PATH} \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16

python collect_step3.py