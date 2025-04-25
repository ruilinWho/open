#######################################################################
# This script is used to build global schema linking data for training.
# Note: Some paths have been anonymized to avoid identity disclosure.
#######################################################################

import json
import random
from pathlib import Path

from sql_metadata import Parser as SQLParser

SPIDER_BASE = Path().cwd().parent / "dataset" / "spider"
BIRD_TRAIN_BASE = Path().cwd().parent / "dataset" / "bird-train"
BIRD_DEV_BASE = Path().cwd().parent / "dataset" / "bird-dev"

TARGET = "spider-train"
# TARGET = "spider-dev"
# TARGET = "bird-train"
# TARGET = "bird-dev"
assert TARGET in ["spider-train", "spider-dev", "bird-train", "bird-dev"]

if TARGET == "spider-train":
    dataset_path1 = Path().cwd().parent / "dataset" / "spider" / "train_spider.json"
    dataset_path2 = Path().cwd().parent / "dataset" / "spider" / "train_others.json"
    train1 = json.load(open(dataset_path1))
    train2 = json.load(open(dataset_path2))
    dataset = train1 + train2
    db_path = SPIDER_BASE / "database"
else:
    if TARGET == "spider-dev":
        dataset_path = Path().cwd().parent / "dataset" / "spider" / "dev.json"
        db_path = SPIDER_BASE / "database"
    elif TARGET == "bird-train":
        dataset_path = Path().cwd().parent / "dataset" / "bird-train" / "train.json"
        db_path = BIRD_TRAIN_BASE / "train_databases"
    elif TARGET == "bird-dev":
        dataset_path = Path().cwd().parent / "dataset" / "bird-dev" / "dev.json"
        db_path = BIRD_DEV_BASE / "dev_databases"
    dataset = json.load(open(dataset_path))


import sqlite3

db_schema_dict = {}
print(db_path)
for db_dir in db_path.iterdir():
    db_id = db_dir.stem
    if db_id == ".DS_Store":
        continue

    ### Parse the DB
    conn = sqlite3.connect(db_dir / f"{db_id}.sqlite")
    cursor = conn.cursor()

    ### Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    schema = {}
    ### For each table, get the columns
    for idx, table_name in enumerate(table_names):
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        columns_with_id = [(id, column[1].lower()) for id, column in enumerate(columns)]

        schema[table_name.lower()] = {"id": idx, "columns": columns_with_id}

    db_schema_dict[db_id] = schema


answer_set = []


from sqlglot import exp, parse_one

for datapoint in dataset:
    if "spider" in TARGET:
        original_query = datapoint["query"]
    else:
        original_query = datapoint["SQL"]

    query = original_query.lower()
    if "is not ''" in query:
        query = query.replace("is not ''", "!= ''")
    if "ref_company_types" in query:
        continue
    if "order details" in query:
        query = query.replace("order details", "orderdetails")

    # print(query)

    schema = db_schema_dict[datapoint["db_id"]]
    # TODO
    if "join" in query and "on" not in query:
        print(query)

    parsed = parse_one(query, dialect="sqlite")

    ### Get the tables and their aliases
    appeared_table_with_columns = {}
    alias_to_sql = {}
    for table in parsed.find_all(exp.Table):
        original_table_name = table.name.lower()
        alias = table.alias

        if original_table_name not in schema:
            continue

        if original_table_name not in appeared_table_with_columns:
            appeared_table_with_columns[original_table_name] = []
        if alias:
            if alias not in alias_to_sql:
                alias_to_sql[alias] = [original_table_name]
            else:
                alias_to_sql[alias].append(original_table_name)

    if "Show name of all students who have some friends and also are liked by someone else." in datapoint["question"]:
        print(alias_to_sql)

    ### Get all the columns
    for column in parsed.find_all(exp.Column):
        ### Two different cases: table exists or not
        if column.table:
            ### Alias or table name

            if column.table in alias_to_sql:
                # table_name = alias_to_sql[column.table] if column.table in alias_to_sql else column.table
                possible_tables = alias_to_sql[column.table]
                for table_name in possible_tables:
                    if table_name not in schema:
                        continue

                    column_names = [column[1] for column in schema[table_name]["columns"]]
                    c_name = column.name.strip()
                    if (c_name in column_names) and (c_name not in appeared_table_with_columns[table_name]):
                        appeared_table_with_columns[table_name].append(c_name)

            else:
                c_name = column.name.strip()
                if (column.table in appeared_table_with_columns) and (c_name not in appeared_table_with_columns[column.table]):
                    appeared_table_with_columns[column.table].append(c_name)

        else:
            ### Search which table has the column
            possible_tables = []
            for table_name in appeared_table_with_columns.keys():
                # XXX: This happens only when CTE is used
                if table_name not in schema:
                    continue

                column_names = [column[1] for column in schema[table_name]["columns"]]
                if column.name in column_names:
                    possible_tables.append(table_name)

            for table_name in possible_tables:
                c_name = column.name.strip()
                if c_name not in appeared_table_with_columns[table_name]:
                    appeared_table_with_columns[table_name].append(c_name)

    bad_dp = False
    schema_link = {}
    for table_name, columns in appeared_table_with_columns.items():
        schema_link[table_name] = columns
        if len(columns) == 0:
            bad_dp = True

    if bad_dp:
        continue

    answer_set.append(
        {
            "db_id": datapoint["db_id"],
            "question": datapoint["question"],
            "gold_query": original_query,
            "query": query,
            "schema_link": schema_link,
        }
    )


with open(f"{TARGET}1.json", "w") as f:
    json.dump(answer_set, f, indent=2)
