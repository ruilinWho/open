import ast
import copy
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import func_timeout
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, SFTConfig, TrlParser

# COT_TEMPLATE_WO_HINT = """\
# For a Text-to-SQL scenario, I will provide you:
# - a database schema
# - a user question
# - two SQL queries: one correct and one incorrect, without indicating which is which.
# - the execution result of the two SQL queries
# Your task is to identify the correct SQL query.
# Please carefully analyze the semantic alignment between the SQL queries and the user's question, and examine the differences between the SQL queries.
# Requirements (in order of priority):
# - (Priority) Judge based on how accurately the SQL answers the user's question
# - If the SQL execution result is empty or None, then this SQL is likely to be incorrect.
# - Using more tables and columns doesn't definitely make an SQL query correct - consider how well it matches the user's question
# - We prefer SQL queries whose results can directly serve as the answer to the user's question
# - If neither SQL is ideal, choose the better one that align with the user's question

# Please analyze and provide output in this specific format:
# 1. Question Analysis
# 2. Semantic and logical observation of SQL 1 (at this stage, avoid discussing potential errors or making conclusions)
# 3. Semantic and logical observation of SQL 2 (at this stage, avoid discussing potential errors or making conclusions)
# 4. Analysis of differences between SQL 1 and SQL 2 (create a detailed Markdown table for comparison)
# 5. Judgement (concluding with either \\box{{SQL1}} or \\box{{SQL2}} as the result).

# [Database Schema]
# {schema}
# [User question]
# {question}
# [SQL 1]
# {sql1}
# [SQL 1 Execution Result]
# {sql1_result}
# [SQL 2]
# {sql2}
# [SQL 2 Execution Result]
# {sql2_result}
# """

# COT_TEMPLATE_WITH_HINT = """\
# For a Text-to-SQL scenario, I will provide you:
# - a database schema
# - a user question
# - a important Hint
# - two SQL queries: one correct and one incorrect, without indicating which is which.
# - the execution result of the two SQL queries
# Your task is to identify the correct SQL query.
# Please carefully analyze the semantic alignment between the SQL queries and the user's question, and examine the differences between the SQL queries.
# Requirements (in order of priority):
# - (Priority) If a Hint specifies the solution approach or provides calculation logic, the SQL query that follows the Hint is more likely to be correct
# - (Priority) Judge based on how accurately the SQL answers the user's question
# - If the SQL execution result is empty or None, then this SQL is likely to be incorrect.
# - Using more tables and columns doesn't definitely make an SQL query correct - consider how well it matches the user's question and Hint
# - We prefer SQL queries whose results can directly serve as the answer to the user's question
# - If neither SQL is ideal, choose the better one based on how well it matches the Hint and the user's question

# Please analyze and provide output in this specific 5-step format:
# 1. Question Analysis
# 2. Semantic and logical observation of SQL 1 (at this stage, avoid discussing potential errors or making conclusions)
# 3. Semantic and logical observation of SQL 2 (at this stage, avoid discussing potential errors or making conclusions)
# 4. Analysis of differences between SQL 1 and SQL 2 (create a detailed Markdown table for comparison)
# 5. Judgement (concluding with either \\box{{SQL1}} or \\box{{SQL2}} as the result).

# [Database Schema]
# {schema}
# [User question]
# {question}
# [Hint]
# {hint}
# [SQL 1]
# {sql1}
# [SQL 1 Execution Result]
# {sql1_result}
# [SQL 2]
# {sql2}
# [SQL 2 Execution Result]
# {sql2_result}
# """


# def generate_COT_prompt(datapoint: dict) -> str:
#     schema = datapoint["schema"]
#     question = datapoint["question"]
#     hint = datapoint["evidence"]
#     sql1 = datapoint["sql1"]
#     sql1_result = datapoint["sql1_result"]
#     sql2 = datapoint["sql2"]
#     sql2_result = datapoint["sql2_result"]

#     if len(hint.strip()) > 4:
#         return COT_TEMPLATE_WITH_HINT.format(
#             schema=schema,
#             question=question,
#             hint=hint,
#             sql1=sql1,
#             sql1_result=sql1_result,
#             sql2=sql2,
#             sql2_result=sql2_result,
#         )
#     else:
#         return COT_TEMPLATE_WO_HINT.format(
#             schema=schema,
#             question=question,
#             sql1=sql1,
#             sql1_result=sql1_result,
#             sql2=sql2,
#             sql2_result=sql2_result,
#         )

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


PAIRWISE_RESPONSE_TEMPLATE = "Response: \n###"


def model_generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 10240,
    response_template: str = PAIRWISE_RESPONSE_TEMPLATE,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_template},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0][input_ids.shape[1] :], skip_special_tokens=True)
    return response


@dataclass
class CustomConfig:
    model_storage_dir: str = field()


parser = TrlParser((CustomConfig, ModelConfig))
(custom_config, model_config) = parser.parse_args_and_config()
custom_config: CustomConfig
model_config: ModelConfig

storage_path = Path.cwd() / "pipeline"
storage_path.mkdir(exist_ok=True)
generation_result = json.load(open("./pipeline/generation_result.json", "r"))


# Load Model
select_model = AutoModelForCausalLM.from_pretrained(
    custom_config.model_storage_dir,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=model_config.torch_dtype,
)
tokenizer = AutoTokenizer.from_pretrained(custom_config.model_storage_dir)

tokenizer.padding_side = "right"

distributed_state = PartialState()
select_model.to(distributed_state.device)

with distributed_state.split_between_processes(generation_result) as generation_data:
    print(f"NCCL_TIMEOUT={os.getenv('NCCL_TIMEOUT')}")
    print(f"device = {distributed_state.device}. Starting to select!")

    subnode_corrects = []
    pbar = tqdm(generation_data, total=len(generation_data), position=distributed_state.process_index)

    for datapoint in pbar:
        candidates: list[dict] = []

        assert len(datapoint["valid_sqls"]) == len(datapoint["valid_results"])
        assert len(datapoint["valid_sqls"]) == len(datapoint["formatted_results"])
        length = len(datapoint["valid_sqls"])
        for i in range(length):
            candidates.append(
                {
                    "sql": datapoint["valid_sqls"][i],
                    "format_result": datapoint["formatted_results"][i],
                    "full_result": ast.literal_eval(datapoint["valid_results"][i]),
                }
            )

        # result_memory = [ast.literal_eval(s) for s in datapoint["valid_results"]]

        match length:
            case 0:
                selected_result = [(0,)]
                selected_sql = "SELECT 0"
                wins = []
            case 1:
                selected_result = candidates[0]["full_result"]
                selected_sql = candidates[0]["sql"]
                wins = []
            case _:
                # Select the best result
                while len(candidates) > 1:
                    next_candidates = []
                    random.shuffle(candidates)
                    # compare (i, i + 1)
                    for i in range(0, len(candidates), 2):
                        # if this is the last one, add it to the next round
                        if i == len(candidates) - 1:
                            next_candidates.append(candidates[i])
                            continue
                        if set(candidates[i]["full_result"]) == set(candidates[i + 1]["full_result"]):
                            next_candidates.append(candidates[i])
                            continue

                        prompt = generate_COT_prompt(
                            {
                                "schema": datapoint["refined_schema"],
                                "question": datapoint["question"],
                                # "evidence": datapoint["evidence"],
                                "sql1": candidates[i]["sql"],
                                "sql1_results": candidates[i]["format_result"],
                                "sql2": candidates[i + 1]["sql"],
                                "sql2_results": candidates[i + 1]["format_result"],
                            }
                        )
                        response = model_generate_single(select_model, tokenizer, prompt)

                        l1 = r"\box{SQL1}" in response
                        l2 = r"\box{SQL2}" in response


                        if not ((l1 and not l2) or (l2 and not l1)):
                            next_candidates.append(candidates[i])
                        elif l1:
                            next_candidates.append(candidates[i])
                        else:
                            next_candidates.append(candidates[i + 1])

                    # Round end
                    candidates = next_candidates

                selected_sql = candidates[0]["sql"]
                selected_result = candidates[0]["full_result"]

        if not all(type(row) is tuple for row in selected_result):
            with open("./pipeline/bad_select_result.txt", "w") as f:
                print("selected_result:", selected_result)
                f.write(selected_result)

        corrct_result = ast.literal_eval(datapoint["gold_result"])
        assert type(selected_result) is list and all(type(row) is tuple or type(row) is str for row in selected_result)
        assert type(corrct_result) is list and all(type(row) is tuple or type(row) is str for row in corrct_result)
        compare_result = set(selected_result) == set(corrct_result)

        res = copy.deepcopy(datapoint)
        res["selected_sql"] = selected_sql
        res["selected_result"] = selected_result
        res["selected_correct"] = compare_result
        subnode_corrects.append(res)

    with open(f"./pipeline/res_{distributed_state.process_index}.json", "w") as f:
        json.dump(subnode_corrects, f, indent=2)
