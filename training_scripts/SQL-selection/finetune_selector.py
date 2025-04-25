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

Response_template = "Response: \n"


@dataclass
class CustomConfig:
    cache_dir: str = field()
    base_path: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()


COT_TEMPLATE_WO_HINT = """
For a Text-to-SQL scenario, I will provide you:
- a database schema
- a user question
- two SQL queries: one correct and one incorrect, without indicating which is which.
- the execution result of the two SQL queries.
Your task is to identify the correct SQL query.

[Database Schema]
{schema}
[User question]
{question}
[SQL 1]
{sql1}
[SQL 1 Execution Results]
{sql1_results}
[SQL 2]
{sql2}
[SQL 2 Execution Results]
{sql2_results}
"""

COT_TEMPLATE_WO_HINT = """\
For a Text-to-SQL scenario, I will provide you:
- a database schema
- a user question
- two SQL queries: one correct and one incorrect, without indicating which is which.
- the execution results of the SQL queries.
Task: ** Your task is to identify the correct SQL query. **
Please carefully analyze the semantic alignment between the SQL queries and the user's question, and examine the differences between the SQL queries.

Requirements:
- Judge based on how accurately the SQL answers the user's question, the output schema should match the user's question: not provide redundant columns, nor omit any necessary columns.
- The more complex SQL query is not necessarily the correct one.
- Using more tables and columns doesn't definitely make an SQL query correct.
- The execution results may help you identify the correct SQL, but empty results does not mean a SQL is wrong, the main focus is on the SQL query itself.
- ** If both SQL queries are correct / incorrect, please choose the better SQL query. **

Please analyze and provide output in this specific format:
1. Question Analysis
2. Semantic and logical observation of SQL 1 (at this stage, avoid discussing potential errors or making conclusions)
3. Semantic and logical observation of SQL 2 (at this stage, avoid discussing potential errors or making conclusions)
4. Analysis of differences between SQL 1 and SQL 2
    - note: Observe different details in the SQL queries
    - Analyze the differences of the execution results and think about why the results are different. This may help you identify the correct SQL query.
5. Judgement (concluding with either \\box{{SQL1}} or \\box{{SQL2}} as the result).

[Database Schema]
{schema}
[User question]
{question}
[SQL 1]
{sql1}
[SQL 1 Execution Results]
{sql1_results}
[SQL 2]
{sql2}
[SQL 2 Execution Results]
{sql2_results}
"""


def generate_COT_prompt(datapoint: dict) -> str:
    schema = datapoint["schema"]
    question = datapoint["question"]
    sql1 = datapoint["sql1"]
    sql1_results = datapoint["sql1_results"]
    sql2 = datapoint["sql2"]
    sql2_results = datapoint["sql2_results"]

    return COT_TEMPLATE_WO_HINT.format(
        schema=schema,
        question=question,
        sql1=sql1,
        sql1_results=sql1_results,
        sql2=sql2,
        sql2_results=sql2_results,
    )


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts_colletion = []
    dataset_size = len(training_dataset["prompt"])
    for i in range(dataset_size):
        input_prompt = training_dataset["prompt"][i]
        output = Response_template + training_dataset["response"][i].strip("\n")

        messages = [
            {"role": "user", "content": input_prompt},
            {"role": "assistant", "content": output},
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

    # Wandb login and init
    if "32B" in model_config.model_name_or_path:
        model_name = "32B"
    elif "7B" in model_config.model_name_or_path:
        model_name = "7B"
    elif "8B" in model_config.model_name_or_path:
        model_name = "8B"
    else:
        raise ValueError("Model name must be 32B or 7B")

    lr = str(training_args.learning_rate)

    ####################################
    # 1. Model Init kwargs & Tokenizer #
    ####################################
    print("##### Model dir:", model_config.model_name_or_path)
    print("##### Cache dir:", custom_config.cache_dir)
    base_gen_path = custom_config.base_path

    model = AutoModelForCausalLM.from_pretrained(
        base_gen_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
    )

    print("###### Model device map:", model.device)
    print("###### Using distributed training:", torch.distributed.is_initialized())

    #################################
    # 2. Load & Tweak the tokenizer #
    #################################
    tokenizer = AutoTokenizer.from_pretrained(base_gen_path)
    tokenizer.padding_side = "right"

    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    print("##### All train data size:", len(finetune_data_json))

    train_data = Dataset.from_list(finetune_data_json)
    train_data = train_data.shuffle()

    ###############
    # 4. Training #
    ###############
    collator = DataCollatorForCompletionOnlyLM(Response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=train_data,
    )
    trainer.train()

    ###########################
    # 5. Save the final model #
    ###########################
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name}_lr{lr}"
    print("##### Storage Path:", storage_path)

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
