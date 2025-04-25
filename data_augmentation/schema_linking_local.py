#######################################################################
# This script is used to build local schema linking data for training.
# Note: Some paths have been anonymized to avoid identity disclosure.
#######################################################################

import json
import pickle
import random
from pathlib import Path

from embed_values import ColumnVectorIndex
from ir_to_schema import IR2Schema
from sentence_transformers import SentenceTransformer

TRAIN_IR_PATH = Path("./irs/bird_dev_ir.json")
SCHEMA_LINK_PATH = Path("./schema_link_evidence/bird-dev.json")
INDEX_DIR = Path("/home/Anonymous/indexes/bird-dev")

ir_set = json.load(TRAIN_IR_PATH.open())
schema_link = json.load(SCHEMA_LINK_PATH.open())
print(len(ir_set), len(schema_link))

index_dict = {}
db_ids = [db_id.stem for db_id in Path(INDEX_DIR).glob("*.pkl")]
for db_id in db_ids:
    index_path = Path(INDEX_DIR) / f"{db_id}.pkl"
    index = pickle.load(open(index_path, "rb"))
    index_dict[db_id] = index
print(len(index_dict))


DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
emb_model = SentenceTransformer(DEFAULT_MODEL, device="cuda:2", trust_remote_code=True)


"""
Prepare for a local classification training data.
Input: Schema, Question, a column
Output: Whether this column is useful for the question (True/False)
"""

LOCAL_CLASSIFICATION_TEMPLATE = """Given a database table, a question, and a column in the table, your task is to determine whether the column is useful to generate a SQL query for answering the question.
Note: Some example values of the column are shown to you, if any example values match the question, the column is likely to be useful.

[Table schema]
{table_schema}
[Column to check]
column name: {column_name}
{column_value_examples}
[Question]
{question}

-- Return one word: True or False.
"""


from tqdm import tqdm

total_dataset = []

for datapoint in tqdm(schema_link, total=len(schema_link)):
    db_id = datapoint["db_id"]
    question = datapoint["question"]
    evidence = datapoint["evidence"]

    if len(evidence.strip()) >= 5:
        question = question + "\n" + "hint: " + evidence

    ir = [ir for ir in ir_set if ir["db_id"] == db_id][0]
    index = index_dict[db_id]
    converter = IR2Schema(ir, None, index, question, emb_model, None)

    links: list[dict] = datapoint["schema_link"]
    for link in links:
        table_name = link["table_name"]

        table_ir = [t for t in ir["tables"] if t["table_name"] == table_name][0]

        # get all column_names in this table (not including primary keys and foreign keys)
        all_columns = [col["col_name"] for col in table_ir["columns"] if col["col_idx"] not in table_ir["primary_keys"]]
        for foreign_key in table_ir["foreign_keys"]:
            fk_table_name = foreign_key["table"].strip('"')
            fk_column_name = foreign_key["column"].strip('"')

            if (fk_table_name == table_name) and (fk_column_name in all_columns):
                all_columns.remove(fk_column_name)

        # Now all_columns only contains columns that are not PK and not FK
        used_columns = link["related_columns"]
        not_used_columns = [col for col in all_columns if col not in used_columns]

        # sample 1:1
        sample_num = min(len(used_columns), len(not_used_columns))
        used_columns = random.sample(used_columns, sample_num)
        not_used_columns = random.sample(not_used_columns, sample_num)

        for column_name in used_columns:
            table_schema, column_value_examples = converter.get_specific_schema(table_name, column_name)
            prompt = LOCAL_CLASSIFICATION_TEMPLATE.format(
                table_schema=table_schema,
                column_name=column_name,
                column_value_examples=column_value_examples,
                question=question,
            )
            total_dataset.append({"prompt": prompt, "label": "True"})

        for column_name in not_used_columns:
            table_schema, column_value_examples = converter.get_specific_schema(table_name, column_name)
            prompt = LOCAL_CLASSIFICATION_TEMPLATE.format(
                table_schema=table_schema,
                column_name=column_name,
                column_value_examples=column_value_examples,
                question=question,
            )
            total_dataset.append({"prompt": prompt, "label": "False"})

print(len(total_dataset))


with open("./bird_dev.json", "w") as f:
    json.dump(total_dataset, f, indent=2)
