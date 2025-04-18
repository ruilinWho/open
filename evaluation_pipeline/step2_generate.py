import copy
import json
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

### Constants
DB_BASE = Path("/home/ubuntu/database_files/bird-dev/dev_databases")


@dataclass
class CustomConfig:
    model_storage_dir: str = field()
    refined_input_path: str = field()


@func_timeout.func_set_timeout(30)
def exec_on_db(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        conn.close()


def format_result(r: str | list):
    if type(r) is str and r[:6] == "Error:":
        return r

    result = copy.deepcopy(r)

    if len(result) == 0:
        formatted = "The execution result is empty."
    elif len(result) <= 10:
        formatted = f"The execution result contains {len(result)} rows.\n"
        for i in range(len(result) - 1, -1, -1):
            if len(str(result[i])) > 200:
                result[i] = str(result[i])[:200] + "..."
        formatted += "\n".join([str(row) for row in result])
    else:
        original_len = len(result)
        result = result[:10]
        for i in range(len(result) - 1, -1, -1):
            if len(str(result[i])) > 200:
                result[i] = str(result[i])[:200] + "..."

        formatted = f"The execution result contains {original_len} rows. (showing the top 10 rows)\n"
        formatted += "\n".join([str(row) for row in result])
    return formatted


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


# def extract_sql(s: str) -> str:
#     # extract from ```sql and ```
#     pattern = r"```sql\s*(.*?)\s*```"
#     match = re.search(pattern, s, re.DOTALL)

#     s_1 = match.group(1) if match else s

#     # Try to extract WITH first
#     pattern_with = r"(WITH.*)"
#     match_with = re.search(pattern_with, s_1, re.DOTALL)
#     if match_with:
#         s_2 = match_with.group(1)
#     else:
#         # If no WITH match, try to extract SELECT
#         pattern_select = r"(SELECT.*)"
#         match_select = re.search(pattern_select, s_1, re.DOTALL)
#         s_2 = match_select.group(1) if match_select else s_1

#     # Remove all the \t, \n, and continuous spaces
#     s_3 = re.sub(r"\s+", " ", s_2).strip()

#     return s_3


def extract_sqls_to_list(s: str) -> list[str]:
    # extract from ```sql and ```
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.finditer(pattern, s, re.DOTALL)
    sql_statements = []

    for match in matches:
        sql_block = match.group(1)

        # 处理WITH开头的语句
        pattern_with = r"(WITH.*)"
        match_with = re.search(pattern_with, sql_block, re.DOTALL)
        if match_with:
            s_2 = match_with.group(1)
        else:
            # 如果没有WITH，尝试提取SELECT
            pattern_select = r"(SELECT.*)"
            match_select = re.search(pattern_select, sql_block, re.DOTALL)
            s_2 = match_select.group(1) if match_select else sql_block

        # 清理多余的空白字符
        s_3 = re.sub(r"\s+", " ", s_2).strip()

        if s_3:  # 确保提取的SQL不为空
            sql_statements.append(s_3)
    return sql_statements


def nl2sql_generate_multiple(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, num: int) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": NL2SQL_RESPONSE_TEMPLATE + "1. ```sql\n"},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    device = model.device
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        # temperature=0.5,
        temperature=1.2,
        num_return_sequences=num,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = []
    for output in outputs:
        responses.append(tokenizer.decode(output[input_ids.shape[1] :], skip_special_tokens=True))
    return responses


def nl2sql_generate_beam(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num: int,
    group: int,
    penalty: int,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": NL2SQL_RESPONSE_TEMPLATE + "1. ```sql\n"},
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    device = model.device
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", padding=False).to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=False,
        num_beams=num,
        num_return_sequences=num,
        num_beam_groups=group,
        diversity_penalty=penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = []
    for output in outputs:
        responses.append(tokenizer.decode(output[input_ids.shape[1] :], skip_special_tokens=True))
    return responses


##################################################
parser = TrlParser((CustomConfig, ModelConfig))
(custom_config, model_config) = parser.parse_args_and_config()
custom_config: CustomConfig
model_config: ModelConfig

storage_path = Path.cwd() / "pipeline"
storage_path.mkdir(exist_ok=True)
refined_input = json.load(open(custom_config.refined_input_path, "r"))

model = AutoModelForCausalLM.from_pretrained(
    custom_config.model_storage_dir,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=model_config.torch_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(custom_config.model_storage_dir)
tokenizer.padding_side = "right"

distributed_state = PartialState()
model.to(distributed_state.device)

with distributed_state.split_between_processes(refined_input) as refined_data:
    print(f"device = {distributed_state.device}. Starting to validate!")

    subnode_results = []
    pbar = tqdm(refined_data, total=len(refined_data), position=distributed_state.process_index)
    for datapoint in pbar:
        # XXX Use Prepared Pruned Schema (do not generate pruned schema in this file.)
        pruned_schema = datapoint["refined_schema"]

        # Use Pruned Schema to generate NL2SQL Prompt
        nl2sql_prompt = generate_NL2SQL_prompt(
            {
                "schema": pruned_schema,
                "question": datapoint["question"],
                "evidence": datapoint["evidence"],
            }
        )

        responses1 = nl2sql_generate_multiple(nl2sql_prompt, model, tokenizer, 6)
        responses2 = nl2sql_generate_beam(nl2sql_prompt, model, tokenizer, 6, 6, 1.1)
        responses = responses1 + responses2

        gold_sql = datapoint["query"]
        pred_sqls = []
        for response in responses:
            pred_sql = extract_sqls_to_list("1. ```sql\n" + response)
            pred_sqls.extend(pred_sql)

        # print("Pred_sqls: ", pred_sqls)

        db_id = datapoint["db_id"]
        db_path = DB_BASE / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(gold_sql)
        try:
            gold_result = cur.fetchall()
        except:
            gold_result = []

        # Do Self-Consistency
        exe_results: dict[str, list[str]] = {}
        exe_to_formated = {}

        able_to_correct = False
        for pred_sql in pred_sqls:
            try:
                pred_result = exec_on_db(db_path, pred_sql)

                if set(pred_result) == set(gold_result):
                    able_to_correct = True
                s_result = str(pred_result)

                if s_result not in exe_to_formated:
                    exe_to_formated[s_result] = format_result(pred_result)

                if s_result not in exe_results:
                    exe_results[s_result] = [pred_sql]
                else:
                    exe_results[s_result].append(pred_sql)
            except Exception:
                # print("Fail:", pred_sql, e)
                continue
            except func_timeout.FunctionTimedOut:
                continue

        valid_sqls = []
        valid_results = []
        weights = []
        formatted_results = []

        if len(exe_results) == 0:
            final_sql = "SELECT 0;"
            best_result = [(0,)]
        else:
            # Choose the one with most frequency
            max_frequency = max([len(v) for v in exe_results.values()])
            best_result = [res for res, sqls in exe_results.items() if len(sqls) == max_frequency][0]
            final_sql = random.choice(exe_results[best_result])

            # at every result, choose one
            for res, sqls in exe_results.items():
                for sql in sqls:
                    valid_sqls.append(sql)
                    valid_results.append(res)
                    weights.append(len(sqls))
                    formatted_results.append(exe_to_formated[res])

        cur.execute(final_sql)
        pred_result = cur.fetchall()
        conn.close()

        result = copy.deepcopy(datapoint)

        # add more key
        # result["pred_sqls"] = pred_sqls
        result["consistency_sql"] = final_sql
        result["consistency_result"] = best_result
        result["consistency_correct"] = set(pred_result) == set(gold_result)
        result["able_to_correct"] = able_to_correct
        result["gold_result"] = str(gold_result)
        result["valid_sqls"] = valid_sqls
        result["formatted_results"] = formatted_results
        result["valid_results"] = valid_results
        result["weights"] = weights

        subnode_results.append(result)

    result_store_path = storage_path / f"GPU_{distributed_state.process_index}_results.json"
    with open(result_store_path, "w") as f:
        json.dump(subnode_results, f, indent=2)
