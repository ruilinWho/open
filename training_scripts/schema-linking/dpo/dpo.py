import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
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
    model_storage_dir: str = field()
    finetune_data_dir: str = field()
    sft_base_path: str = field()


def build_formatted_DPO_data(raw_dpo_data: list[dict], tokenizer: AutoTokenizer) -> Dataset:
    data_list = []
    for datapoint in raw_dpo_data:
        if not datapoint["evidence"].strip():
            hint = ""
        else:
            hint = f"\n### Hint: {datapoint['evidence']}"
        prompt = SCHEMA_LINK_TEMPLATE.format(schema=datapoint["schema"], question=datapoint["question"], hint=hint)
        data_list.append(
            {
                "prompt": prompt,
                "chosen": SCHEMA_LINK_RESPONSE_TEMPLATE + datapoint["win_sl"],
                "rejected": SCHEMA_LINK_RESPONSE_TEMPLATE + datapoint["loss_sl"],
            }
        )
    return Dataset.from_list(data_list).shuffle()


if __name__ == "__main__":
    ##################
    # Parse argument #
    ##################
    parser = TrlParser((CustomConfig, ModelConfig, DPOConfig))
    (custom_config, model_config, dpo_config) = parser.parse_args_and_config()
    custom_config: CustomConfig
    model_config: ModelConfig
    dpo_config: DPOConfig

    if "7B" in model_config.model_name_or_path:
        model_name = "7B"
    elif "8B" in model_config.model_name_or_path:
        model_name = "8B"
    elif "14B" in model_config.model_name_or_path:
        model_name = "14B"
    elif "32B" in model_config.model_name_or_path:
        model_name = "32B"

    lr = str(dpo_config.learning_rate)
    beta = str(dpo_config.beta)
    rpo_alpha = str(dpo_config.rpo_alpha)
    parameter_summary = f"{model_name}_lr_{lr}_beta_{beta}_rpo_alpha_{rpo_alpha}"

    ####################################
    # 1. Model Init kwargs & Tokenizer #
    ####################################
    print("##### Model dir:", model_config.model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        custom_config.sft_base_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
    )

    # Create Reference Model same as model
    reference_model = AutoModelForCausalLM.from_pretrained(
        custom_config.sft_base_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
    )

    print("###### Model device map:", model.device)
    print("###### Using distributed training:", torch.distributed.is_initialized())

    #################################
    # 2. Load & Tweak the tokenizer #
    #################################
    tokenizer = AutoTokenizer.from_pretrained(custom_config.sft_base_path)
    tokenizer.padding_side = "left"

    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    dpo_finetune_data = [d for d in finetune_data_json if d["train_type"] == "DPO"]
    reg_dpo = []
    for dp in dpo_finetune_data:
        full_text = SCHEMA_LINK_TEMPLATE.format(schema=dp["schema"], hint=dp["evidence"], question=dp["question"]) + dp["win_sl"]
        if len(tokenizer.encode(full_text)) <= dpo_config.max_length:
            reg_dpo.append(dp)

    print(f"##### DPO Datasize From {len(dpo_finetune_data)} --> {len(reg_dpo)}")
    dpo_dataset = build_formatted_DPO_data(reg_dpo, tokenizer)

    ######################
    # 5. DPO fine-tuning #
    ######################
    trainer = DPOTrainer(
        model=model,
        # ref_model=None,
        ref_model=reference_model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    ######################
    # 6. Store the model #
    ######################
    storage_path = Path(custom_config.model_storage_dir) / parameter_summary
    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
