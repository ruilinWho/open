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

NL2SQL_TEMPLATE_WO_HINT = """[Task] 
Given a database schema, a user's natural language question, your task is to generate SQLite queries to answer the user question.

[Task Requirements]
- If a user question can be solved by multiple equivalent SQL queries (e.g. Multi-table JOIN, Common Table Expression, Subqueries), please output all these equivalent SQL queries.
- When outputting multiple SQL queries, please ensure that their syntactic structures differ while maintaining the same semantic meaning. Also, avoid generating overly complex or unreasonable queries.
- The SQLite queries should be valid and executable in the given database schema.
- The queries should be semantically aligned with the user question.
- By executing the queries, the database should be able to return the correct answer to the user question.

[Points to note]
- If there is a one-to-many JOIN and you think duplicate rows may lead to incorrect answers, please consider using DISTINCT.
- If a column may be empty, please consider using IS NOT NULL.
- The example values are important. If possible, first consider using matched example values, then consider using values in the hint.

[Database Schema]
{schema}
[Natural Language Question]
{question}
"""

NL2SQL_TEMPLATE_WITH_HINT = """[Task] 
Given a database schema, a user's natural language question, your task is to generate SQLite queries to answer the user question.

[Task Requirements]
- If a user question can be solved by multiple equivalent SQL queries (e.g. Multi-table JOIN, Common Table Expression, Subqueries), please output all these equivalent SQL queries.
- When outputting multiple SQL queries, please ensure that their syntactic structures differ while maintaining the same semantic meaning. Also, avoid generating overly complex or unreasonable queries.
- The SQLite queries should be valid and executable in the given database schema.
- The queries should be semantically aligned with the user question.
- By executing the queries, the database should be able to return the correct answer to the user question.

[Points to note]
- If there is a one-to-many JOIN and you think duplicate rows may lead to incorrect answers, please consider using DISTINCT.
- If a column may be empty, please consider using IS NOT NULL.
- The example values are important. If possible, first consider using matched example values, then consider using values in the hint.

[Database Schema]
{schema}
[Natural Language Question]
{question}
[Hint]
Pay attention to the following hint, which can help you generate the correct SQL queries.
** {hint} **
"""

NL2SQL_RESPONSE_TEMPLATE = "[SQL Query]\n"


def generate_NL2SQL_prompt(datapoint: dict) -> str:
    schema = datapoint["schema"]
    question = datapoint["question"]
    hint = datapoint["evidence"]

    if len(hint.strip()) > 5:
        prompt = NL2SQL_TEMPLATE_WITH_HINT.format(schema=schema, question=question, hint=hint)
    else:
        prompt = NL2SQL_TEMPLATE_WO_HINT.format(schema=schema, question=question)
    return prompt


@dataclass
class CustomConfig:
    cache_dir: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()
    clean_long_data: bool = field()


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts_colletion = []
    dataset_size = len(training_dataset["schema"])

    for i in range(dataset_size):
        if not training_dataset["evidence"][i].strip():
            hint = ""
        else:
            hint = f"\n### Hint: {training_dataset['evidence'][i]}"

        input_prompt: str = generate_NL2SQL_prompt(
            {
                "schema": training_dataset["schema"][i],
                "question": training_dataset["question"][i],
                "evidence": training_dataset["evidence"][i],
            }
        )

        sql_queries = ""
        for j, sql in enumerate(training_dataset["correct_sqls"][i]):
            sql_queries += f"{j+1}. "
            sql_queries += "```sql\n" + sql + "```\n"

        output_sequence: str = NL2SQL_RESPONSE_TEMPLATE + sql_queries + tokenizer.eos_token
        messages = [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": output_sequence},
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
    
    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    print("##### All data size:", len(finetune_data_json))
    if custom_config.clean_long_data:

        def not_too_long(x: dict):
            full_text = generate_NL2SQL_prompt(
                {
                    "schema": x["schema"],
                    "question": x["question"],
                    "evidence": x["evidence"],
                }
            )
            return len(tokenizer.encode(full_text)) <= training_args.max_seq_length - 64

        finetune_data_json = [datapoint for datapoint in finetune_data_json if not_too_long(datapoint)]
        print("##### Filtered data size:", len(finetune_data_json))
    finetune_data = Dataset.from_list(finetune_data_json).shuffle()

    ###############
    # 4. Training #
    ###############
    collator = DataCollatorForCompletionOnlyLM(NL2SQL_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=finetune_data,
    )
    trainer.train()

    ####################################
    # 5. Validation for the first time #
    ####################################
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name}_lr{lr}"
    print("##### Storage Path:", storage_path)

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
