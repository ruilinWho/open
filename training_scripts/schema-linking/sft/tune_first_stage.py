"""
This file trains a causal LLM to do schema-linking.
Firstly do SFT to train a schema-linking model.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

SCHEMA_LINK_TEMPLATE = """Given the following SQL tables, your job is to determine the column names and table that the question is referring to. \
The selected tables and columns should be sufficient to construct a SQL query that answers the question. \
Output in json format to indicate which table uses which column.
{schema}{hint}
### Question: {question}
"""

SCHEMA_LINK_RESPONSE_TEMPLATE = "### Chosen tables and columns:\n"


@dataclass
class CustomConfig:
    cache_dir: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()
    validation_data_dir: str = field()
    clean_long_data: bool = field()


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts_colletion = []
    dataset_size = len(training_dataset["schema"])

    # print("Dataset_size: ", dataset_size)

    for i in range(dataset_size):
        if not training_dataset["evidence"][i].strip():
            hint = ""
        else:
            hint = f"\n### Hint: {training_dataset['evidence'][i]}"

        input_prompt: str = SCHEMA_LINK_TEMPLATE.format(
            schema=training_dataset["schema"][i],
            question=training_dataset["question"][i],
            hint=hint,
        )

        sql: str = SCHEMA_LINK_RESPONSE_TEMPLATE + training_dataset["standard_sl"][i]
        messages = [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": sql},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts_colletion.append(text)
    return texts_colletion


if __name__ == "__main__":
    ##################
    # Parse argument #
    ##################
    parser = TrlParser((CustomConfig, ModelConfig, SFTConfig))
    (custom_config, model_config, training_args) = parser.parse_args_and_config()
    custom_config: CustomConfig
    model_config: ModelConfig
    training_args: SFTConfig

    training_args_dict = training_args.to_dict()

    # Wandb login and init
    current_time = datetime.now().strftime("%m%d-%H-%M")

    if "7B" in model_config.model_name_or_path:
        model_name = "7B"
    elif "8B" in model_config.model_name_or_path:
        model_name = "8B"
    elif "14B" in model_config.model_name_or_path:
        model_name = "14B"
    elif "32B" in model_config.model_name_or_path:
        model_name = "32B"

    lr = str(training_args.learning_rate)
    # wandb.init(project=f"Schema_Link_{model_name}", name=f"lr_{lr}")

    ####################################
    # 1. Model Init kwargs & Tokenizer #
    ####################################
    print("##### Model dir:", model_config.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        cache_dir=custom_config.cache_dir,
    )

    print("###### Model device map:", model.device)
    print("###### Using distributed training:", torch.distributed.is_initialized())

    #################################
    # 2. Load & Tweak the tokenizer #
    #################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
    )
    tokenizer.padding_side = "right"
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    # tokenizer.padding_side = "left"

    

    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    sft_finetune_data = [d for d in finetune_data_json if d["train_type"] == "SFT"]

    print("##### All data size:", len(finetune_data_json))
    if custom_config.clean_long_data:
        # Regularize SFT data
        reg_sft = []
        for dp in sft_finetune_data:
            full_text = SCHEMA_LINK_TEMPLATE.format(schema=dp["schema"], hint=dp["evidence"], question=dp["question"]) + dp["standard_sl"]
            if len(tokenizer.encode(full_text)) <= training_args.max_seq_length:
                reg_sft.append(dp)
        print(f"##### SFT Datasize From {len(sft_finetune_data)} --> {len(reg_sft)}")
        sft_finetune_data = reg_sft

    sft_dataset = Dataset.from_list(sft_finetune_data).shuffle()

    ###############
    # 4. Training #
    ###############
    collator = DataCollatorForCompletionOnlyLM(SCHEMA_LINK_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=sft_dataset,
    )
    trainer.train()

    ####################################
    # 5. Validation for the first time #
    ####################################
    current_time = datetime.now().strftime("%m%d-%H-%M")
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name}_lr{lr}"
    print("##### Storage Path:", storage_path)

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
