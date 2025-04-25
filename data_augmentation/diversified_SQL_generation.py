#######################################################################
# This script is used to build diversified SQL generation data for training.
# It generates multiple SQL queries for each data point in the dataset.
# Note: Some paths have been anonymized to avoid identity disclosure.
#######################################################################

import concurrent.futures
import copy
import functools
import json
import multiprocessing
import os
import random
import re
import signal
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import func_timeout
from tqdm import tqdm

data = json.load(open("./bird-train-dynamic-short.json"))

all_db_ids = list(set(dp["db_id"] for dp in data))

DB_BASE = Path("<Anonymous>")
import openai


def retry_with_exponential_backoff(
    errors: tuple,
    initial_delay: float = 10,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 6,
):
    """
    Retry a function with exponential backoff.
    :param errors: Tuple of errors to retry on.
    :param initial_delay: Initial delay in seconds.
    :param exponential_base: Base for exponential backoff.
    :param jitter: Add jitter to the delay.
    :param max_retries: Maximum number of retries.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.") from None
                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    # Sleep for the delay
                    time.sleep(delay)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


OPENAI_POTENTIAL_ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@retry_with_exponential_backoff(OPENAI_POTENTIAL_ERRORS)
def openai_completion_with_backoff(*args, **kwargs) -> str:
    """Retry the OpenAI completion call with exponential backoff."""
    assert OPENAI_CLIENT is not None
    return openai_completion(*args, **kwargs)


def openai_completion(
    prompt: str,
    model: str = "deepseek-r1-250120",
    temperature: float = 0.2,
    top_p: float = 1,
    max_tokens: int = 8192,
) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def extract_sql(s: str) -> str:
    # extract from ```sql and ```
    pattern = r"```sql\s*(.*?)\s*```"
    match = re.search(pattern, s, re.DOTALL)

    s_1 = match.group(1) if match else s

    # Try to extract WITH first
    pattern_with = r"(WITH.*)"
    match_with = re.search(pattern_with, s_1, re.DOTALL)
    if match_with:
        s_2 = match_with.group(1)
    else:
        # If no WITH match, try to extract SELECT
        pattern_select = r"(SELECT.*)"
        match_select = re.search(pattern_select, s_1, re.DOTALL)
        s_2 = match_select.group(1) if match_select else s_1

    # Remove all the \t, \n, and continuous spaces
    s_3 = re.sub(r"\s+", " ", s_2).strip()

    return s_3


# @func_timeout.func_set_timeout(30)
# def exec_on_db(db_path, query):
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         cursor.execute(query)
#         return cursor.fetchall()
#     finally:
#         conn.close()


def exec_on_db(db_path, query):
    # 普通的执行，不用 func_timeout
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


def run_query_with_timeout(db_path, query, timeout=30):
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    future = executor.submit(exec_on_db, db_path, query)
    try:
        result = future.result(timeout=timeout)
        return result
    except concurrent.futures.TimeoutError:
        print("TimeoutError: ", query)
        for proc in executor._processes.values():
            try:
                os.kill(proc.pid, signal.SIGTERM)
                print(f"Killed process with PID: {proc.pid}")
            except Exception as e:
                print(f"Failed to kill process {proc.pid}: {e}")
        raise func_timeout.FunctionTimedOut("Query execution timed out.")
    finally:
        executor.shutdown(wait=False)


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


COT_TEMPLATE = """\
For a Text-to-SQL scenario, I will provide you:
- a database schema
- a user question
- some correct SQL queries that can answer the user's question
- some wrong SQL queries that cannot answer the user's question

Your task is to generate a new SQL query to answer the user's question.
Requirements:
- The new SQL query should be a valid SQLite query (do not use too complex SQL syntax)
- The new SQL query should have a different solution approach from the given correct SQL queries.
    - different solution approach means the new SQL query should not be a simple modification of the given correct SQL queries. You should try to change the overall SQL structure, not just change the alias or table order.
    - The new SQL query should be a valid SQL query and return the same result as the given correct SQL queries.
    - for example, you may solve a question with multuple solution approaches: 
        1. multi-table JOIN
        2. common table expressions(CTE) and WITH clause
        3. subqueries
        4. other approaches
- The new SQL query should answer the user's question and return the same result as the given correct SQL queries.
- Your output should generate the same result as the correct output. 
- You should also pay attention to the wrong SQL queries and avoid making the same mistake. (For example if a wrong SQL query times out, you should not generate a similar SQL query that also times out.)
- If you can find a SQL with a different solution approach, return it with ```sql and ``` tags.

Note:
- You should avoid generating SQL queries that are too similar to existing correct SQL queries.
- If you cannot find a SQL with a different solution approach or you think you have to use unnatural syntax, please return "None".
- If you find the given correct SQL queries already cover all reasonable solutions and you have to use inefficient SQL queries to generate a new one, please return "None".
- ** You should focus on generating high-quality SQL queries (suitable for the question) and avoid generating inefficient SQL queries or too verbose SQL queries. **
    - Inefficient SQL examples: (1) Cartesian Product (2) Too deep nested subqueries (3) Improper IN or EXISTS usage (such as EXISTS 1) (4) Subquery with SELECT 1
    - DO NOT generate those inefficient verbose SQL queries. We would rather accept "None".
    - We want elegant and efficient SQL queries that can solve the problem effectively.

[Database Schema]
{schema}
[User question]
{question}
[Correct Output]
{correct_output}
[Correct SQL queries]
{correct_sqls}
[Wrong SQL queries]
{wrong_sqls}
"""


def generate_multiple_sqls(datapoint):
    schema = datapoint["schema"]
    question = datapoint["question"]
    evidence = datapoint["evidence"]

    if len(evidence) > 5:
        question += f"\nHint: {evidence}"

    gt_sql = datapoint["query"]
    if gt_sql[-1] != ";":
        gt_sql += ";"

    correct_sqls = [gt_sql]
    wrong_sqls = []
    timeout_sqls = []

    db_id = datapoint["db_id"]
    db_path = DB_BASE / db_id / f"{db_id}.sqlite"

    try:
        gold_result = run_query_with_timeout(db_path, gt_sql)
    except Exception as e:
        print("@Gold SQL Exception: ", e)
        return None
    except func_timeout.FunctionTimedOut:
        datapoint["correct_sqls"] = [gt_sql]
        datapoint["wrong_sqls"] = []
        return datapoint

    if len(gold_result) == 0:
        datapoint["correct_sqls"] = [gt_sql]
        datapoint["wrong_sqls"] = []
        return datapoint

    for _ in range(5):
        correct_text = ""
        for id, sql in enumerate(correct_sqls):
            correct_text += f"[correct SQL {id + 1}] {sql}\n"
        wrong_text = ""
        for id, sql in enumerate(wrong_sqls):
            wrong_text += f"[wrong SQL {id + 1}] {sql}\n"

        if len(wrong_sqls) == 0:
            wrong_text = "No wrong SQL query yet."

        prompt = COT_TEMPLATE.format(
            schema=schema,
            question=question,
            correct_output=format_result(gold_result),
            correct_sqls=correct_text,
            wrong_sqls=wrong_text,
        )
        # response = openai_completion(prompt)
        response = openai_completion_with_backoff(prompt)

        if "None" in response:
            break

        pred_sql = extract_sql(response)

        try:
            if pred_sql in timeout_sqls:
                continue

            pred_result = run_query_with_timeout(db_path, pred_sql)
            if set(pred_result) == set(gold_result):
                correct_sqls.append(pred_sql)
            else:
                wrong_sqls.append(pred_sql + "(Wrong Result)")
        except Exception as e1:
            wrong_sqls.append(pred_sql + f"(Exception: {e1})")
            continue
        except func_timeout.FunctionTimedOut:
            wrong_sqls.append(pred_sql + "(Timeout)")
            timeout_sqls.append(pred_sql)
            continue

        if len(correct_sqls) >= 3:
            break

    # add "correct_sqls" and "wrong_sqls" to datapoint
    datapoint["correct_sqls"] = correct_sqls
    datapoint["wrong_sqls"] = wrong_sqls

    return datapoint


if __name__ == "__main__":
    all_cot_data = []
    cot_data_lock = threading.Lock()

    # Save Directory
    save_dir = Path("./test")
    save_dir.mkdir(exist_ok=True)

    result_count = 0

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(generate_multiple_sqls, datapoint) for datapoint in data[100:105]]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            result_count += 1
            print(f"Result Count: {result_count}")
            if result is not None:
                with cot_data_lock:
                    all_cot_data.append(result)

                    if len(all_cot_data) % 100 == 0:
                        with open(save_dir / f"spider-train-dynamic-short-multiple_{len(all_cot_data)}.json", "w") as f:
                            json.dump(all_cot_data, f, indent=2)

    # Save results
    with open(save_dir / "test.json", "w") as f:
        json.dump(all_cot_data, f, indent=2)
