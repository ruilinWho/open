import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

LOCAL_RESPONSE_TEMPLATE = "[Judgement]\n"


@dataclass
class CustomConfig:
    cache_dir: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()
    validation_data_dir: str = field()


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts_colletion = []
    dataset_size = len(training_dataset["prompt"])
    for i in range(dataset_size):
        input_prompt: str = training_dataset["prompt"][i]
        sql: str = LOCAL_RESPONSE_TEMPLATE + training_dataset["label"][i] + tokenizer.eos_token
        messages = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": sql}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts_colletion.append(text)
    return texts_colletion


def validate_local(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    validation_data_path: str,
    storage_path: Path,
):
    validation_data = json.load(open(validation_data_path, "r"))
    gold_labels = []
    pred_labels = []

    for datapoint in tqdm(validation_data, total=len(validation_data)):
        prompt = datapoint["prompt"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": LOCAL_RESPONSE_TEMPLATE},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(model.device)

        output = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=False)

        gold_labels.append(datapoint["label"] == "True")
        pred_labels.append("True" in response or "true" in response)

    # Calculate precision and recall
    # Precision: if predicted as True, how many are actually True
    # Recall: if actually True, how many are predicted as True
    precision = sum([p and g for p, g in zip(pred_labels, gold_labels)]) / sum(pred_labels)
    recall = sum([p and g for p, g in zip(pred_labels, gold_labels)]) / sum(gold_labels)

    # Write the results
    if not storage_path.exists():
        storage_path.mkdir(parents=True)

    with open(storage_path / "validation_results.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")


if __name__ == "__main__":
    ##################
    # Parse argument #
    ##################
    parser = TrlParser((CustomConfig, ModelConfig, SFTConfig))
    (custom_config, model_config, training_args) = parser.parse_args_and_config()
    custom_config: CustomConfig
    model_config: ModelConfig
    training_args: SFTConfig

    ####################################
    # 1. Model Init kwargs & Tokenizer #
    ####################################
    print("##### Model dir:", model_config.model_name_or_path)
    print("##### Cache dir:", custom_config.cache_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
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

    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    print("##### All data size:", len(finetune_data_json))
    finetune_data = Dataset.from_list(finetune_data_json).shuffle()

    ###############
    # 4. Training #
    ###############
    collator = DataCollatorForCompletionOnlyLM(LOCAL_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=finetune_data,
    )
    trainer.train()

    ###########################
    # 5. Save the final model #
    ###########################
    lr = str(training_args.learning_rate).replace(".", "p")
    storage_path = Path(custom_config.model_storage_dir) / f"3B_lr{lr}"
    print("##### Storage Path:", storage_path)

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True)
    trainer.model.save_pretrained(storage_path)
    tokenizer.save_pretrained(storage_path)

    validate_local(model, tokenizer, custom_config.validation_data_dir, storage_path)